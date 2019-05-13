import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq.modules import MultiheadAttention
from . import register_model, register_model_architecture
from .transformer import *
from fairseq import utils

import argparse

import copy

from collections import OrderedDict

import time

"""
!CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/wiki_ne_en_bpe5000/ --source-lang en \
--target-lang ne --arch cstm_transformer --task cstm_translation --share-all-embeddings \
--encoder-layers 5 --decoder-layers 5 --encoder-embed-dim 512 --decoder-embed-dim 512 \
--encoder-ffn-embed-dim 2048 --decoder-ffn-embed-dim 2048 --encoder-attention-heads 2 \
--decoder-attention-heads 2  --encoder-normalize-before --decoder-normalize-before --dropout 0.4 \
--attention-dropout 0.2 --relu-dropout 0.2 --weight-decay 0.0001 --label-smoothing 0.2 \
--criterion label_smoothed_cross_entropy --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0 \
--lr-scheduler inverse_sqrt --warmup-update 4000 --warmup-init-lr 1e-7 --lr 1e-4 --min-lr 1e-9 \
--max-tokens 2000 --update-freq 4 --max-epoch 100 --save-interval 2 --cstm-share-embeddings \
--cstm-n-retrieved 2 --fp16
"""

"""
In what follows, there is a lot of code that was copied from transformer.py since there
wasn't a great way to wrap it.
"""

class CSTMTransformerDecoderLayer(TransformerDecoderLayer):

	"""
	Decoder transformer block that includes both vanilla attention
	over the encoder output as well as cross attention over the 
	conditional source target memory.
	"""

	def __init__(self, args, no_encoder_attn=False):
		super().__init__(args, no_encoder_attn)

		# this has to be a separate module from normal encoder
		# attention because of the way incremental_state works,
		# but it shares all of its parameters with self.encoder_attn
		self.cstm_attn = MultiheadAttention(
			self.embed_dim, args.decoder_attention_heads,
			dropout=args.attention_dropout
		)
		self.cstm_attn.in_proj_weight = self.encoder_attn.in_proj_weight
		self.cstm_attn.in_proj_bias = self.encoder_attn.in_proj_bias
		self.cstm_attn.out_proj = self.encoder_attn.out_proj
		self.cstm_attn.bias_k = self.encoder_attn.bias_k
		self.cstm_attn.bias_v = self.encoder_attn.bias_v
		self.cstm_attn.add_zero_attn = self.encoder_attn.add_zero_attn

		# feedforward weights for scalar gating variable in cross attention
		self.W_gs = nn.Linear(self.embed_dim, 1, bias=False)
		self.W_gm = nn.Linear(self.embed_dim, 1, bias=False)

	# a lot of this method is copied from transformer.TransformerDecoderLayer
	def forward(self, x, encoder_out, encoder_padding_mask, incremental_state, 
				self_attn_mask=None, self_attn_padding_mask=None):
		residual = x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
		x, _ = self.self_attn(
			query=x,
			key=x,
			value=x,
			key_padding_mask=self_attn_padding_mask,
			incremental_state=incremental_state,
			need_weights=False,
			attn_mask=self_attn_mask,
		)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

		attn = None
		if self.encoder_attn is not None:
			residual = x
			x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
			# attend over vanilla encoder output
			cs, attn = self.encoder_attn(
				query=x,
				key=encoder_out["encoder"],
				value=encoder_out["encoder"],
				key_padding_mask=encoder_padding_mask["encoder"],
				incremental_state=incremental_state,
				static_kv=True,
				need_weights=(not self.training and self.need_attn),
			)
			# attend over conditional source target memory
			if encoder_out["cstm"] is not None:
				cm, attn_cm = self.cstm_attn(
					query=x,
					key=encoder_out["cstm"],
					value=encoder_out["cstm"],
					key_padding_mask=encoder_padding_mask["cstm"],
					incremental_state=incremental_state,
					static_kv=True,
					need_weights=True,
				)
				# gate and combine
				g = (self.W_gs(cs) + self.W_gm(cm)).sigmoid()
				x = g * cs + (1 - g) * cm
			else:
				x = cs
			x = F.dropout(x, p=self.dropout, training=self.training)
			x = residual + x
			x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

		residual = x
		x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, p=self.relu_dropout, training=self.training)
		x = self.fc2(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = residual + x
		x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
		if self.onnx_trace:
			saved_state = self.self_attn._get_input_buffer(incremental_state)
			self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
			return x, attn, self_attn_state
		return x, attn

class CSTMTransformerDecoder(TransformerDecoder):

	"""
	Identical to transformer.TransformerDecoder except it uses the special
	decoder blocks defined above instead of transformer.TransformerDecoderLayer.
	"""

	def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn, left_pad, final_norm)
		self.layers = nn.ModuleList([])
		self.layers.extend([
			CSTMTransformerDecoderLayer(args, no_encoder_attn)
			for _ in range(args.decoder_layers)
		])	

class CSTMTransformerEncoder(TransformerEncoder):

	"""
	First feeds the source sentence through a vanilla transformer encoder. It then
	uses this output to encode retrieved nearest neighbors for the source sentence.
	"""

	def __init__(self, args, src_dict, encoder_embed_tokens, cstm):
		super().__init__(args, src_dict, encoder_embed_tokens)
		self.cstm = cstm

	def forward(self, src_tokens, src_lengths, ids, split):
		encoder_out = super().forward(src_tokens, src_lengths)

		# try to retrieve encoded nearest neighbors
		if len(ids) > 0:
			cstm_out, cstm_padding_mask = self.cstm(encoder_out, ids, split)
		else:
			cstm_out = None 
			cstm_padding_mask = None

		# do this nonsense so that the signature of CSTMTransformerDecoderLayer.forward
		# is the same as the signature of TransformerDecoderLayer.forward,
		# which means we can just use TransformerDecoder.forward for 
		# CSTMTransformerDecoder.forward
		tmp = encoder_out["encoder_out"]
		encoder_out["encoder_out"] = {
			"encoder": tmp,
			"cstm": cstm_out
		}
		tmp = encoder_out["encoder_padding_mask"]
		encoder_out["encoder_padding_mask"] = {
			"encoder": tmp,
			"cstm": cstm_padding_mask
		}
		return encoder_out

	def reorder_encoder_out(self, encoder_out, new_order):
		# for use in beam search
		if encoder_out['encoder_out']['encoder'] is not None:
			encoder_out['encoder_out']['encoder'] = \
				encoder_out['encoder_out']['encoder'].index_select(1, new_order)
		if encoder_out['encoder_out']['cstm'] is not None:
			encoder_out['encoder_out']['cstm'] = \
				encoder_out['encoder_out']['cstm'].index_select(1, new_order)
		if encoder_out['encoder_padding_mask']['encoder'] is not None:
			encoder_out['encoder_padding_mask']['encoder'] = \
				encoder_out['encoder_padding_mask']['encoder'].index_select(0, new_order)
		if encoder_out['encoder_padding_mask']['cstm'] is not None:
			encoder_out['encoder_padding_mask']['cstm'] = \
				encoder_out['encoder_padding_mask']['cstm'].index_select(0, new_order)
		return encoder_out

class CSTM(nn.Module):

	"""
	The class that retrieves and encodes the nearest neighbors for each
	sentence in the batch. 
	"""

	def __init__(self, args, src_dict, src_embed_tokens, \
				 trg_dict, trg_embed_tokens, datasets, nns_data):
		super().__init__()

		new_args = argparse.Namespace(**vars(args))
		new_args.decoder_layers = args.cstm_layers

		self.src_dict = src_dict
		self.trg_dict = trg_dict

		self.retrieved_src_encoder = CSTMInternalEncoder(new_args, src_dict, src_embed_tokens)
		self.retrieved_trg_encoder = CSTMInternalEncoder(new_args, trg_dict, trg_embed_tokens)

		self.n_retrieved = args.cstm_n_retrieved

		self.datasets = copy.deepcopy(datasets)

		self.nns_data = nns_data

	def forward(self, encoder_out, ids, split):
		"""
		This method takes the encoded source sentences as well as their
		ids and which dataset split they come from and retrieves and encodes
		their nearest neighbors.
		"""

		n_retrieved = self.n_retrieved
		# get nearest neighbors as indices
		retrieved = self.retrieve(ids, split, n_retrieved)

		if retrieved is None:
			return None, None

		# reorder encoder_out so that we can encode the retrieved sentences
		# by attending over it
		nns_query_ids = retrieved["nns_query_ids"]
		id_map = torch.tensor([(ids == id).nonzero().item() for id in nns_query_ids])
		tmp = {}
		tmp["encoder_out"] = encoder_out["encoder_out"][:, id_map]
		tmp["encoder_padding_mask"] = \
			encoder_out["encoder_padding_mask"][id_map] \
				if encoder_out["encoder_padding_mask"] is not None else None

		# encode the retrieved source sentences
		retrieved_src_encoding = self.retrieved_src_encoder(
			retrieved["src_tokens"], 
			retrieved["src_padding_mask"],
			tmp
		)

		# encode the retrieved target sentences
		retrieved_trg_encoding = self.retrieved_trg_encoder(
			retrieved["trg_tokens"],
			retrieved["trg_padding_mask"],
			retrieved_src_encoding
		)

		# concatenate all of the retrieved encodings for each
		# source sentence along the seqlen dimension
		trg_enc = retrieved_trg_encoding["encoder_out"]
		trg_pad = retrieved_trg_encoding["encoder_padding_mask"]
		final_trg_enc = []
		final_trg_pad = []
		for idx in ids:
			final_trg_enc.append(
				trg_enc[:, nns_query_ids == idx].reshape(
					retrieved["trg_tokens"].size(1) * n_retrieved,
					-1
				) # (seqlen * n_retrieved) x hidden
			)
			final_trg_pad.append(trg_pad[nns_query_ids == idx].t().flatten())

		final_trg_enc = torch.stack(final_trg_enc).transpose(0, 1)
		final_trg_pad = torch.stack(final_trg_pad)

		test1 = retrieved["trg_tokens"][nns_query_ids == ids[0]]
		self.datasets["valid"].prefetch([int(self.nns_data["train_" + str(ids[0].item())][0].split("_")[1])])
		test2 = self.datasets["valid"][int(self.nns_data["train_" + str(ids[0].item())][0].split("_")[1])]
		print(test2["target"])
		print(final_trg_pad[0].reshape(final_trg_pad.shape[1] // 2, 2).t())

		return final_trg_enc, final_trg_pad

	def retrieve(self, ids, split, n_retrieved):
		# Returns:
		# 	*_tokens: LongTensor of shape (batch * n_retrieved, seqlen)
		# 	*_padding_mask is a ByteTensor of shape (batch * n_retrieved, seqlen) indicating
		# 	 the locations of padding elements
		#
		# Given the source sentence ids and the dataset split to which these sentences
		# belong, this method returns n_retrieved nearest neighbors for each
		# source sentence as a list of BPE indices.

		nns = []
		# Later in the method, we collate all of the nearest neighbors into
		# one big batch. However, Fairseq's collater reorders the batch by
		# length, so we need to keep track of which nearest neighbors belongs
		# to which source sentence in nns_map.
		nns_map = {}
		# for each sentence, retrieve its nearest neighbors
		for idx in ids:
			query_key = split + "_" + str(idx.item())
			nns_keys = self.nns_data[query_key][:n_retrieved]
			# if the sentence doesn't have n_retrieved nearest neighbors,
			# we have to return None, otherwise the model will break (all
			# source sentences must have the same number of retrieved
			# sentences)
			if len(nns_keys) != n_retrieved:
				return None
			nns_keys_train = list(filter(lambda k: "train" in k, nns_keys))
			nns_keys_valid = list(filter(lambda k: "valid" in k, nns_keys))
			self.datasets["train"].prefetch([int(k.split("_")[1]) for k in nns_keys_train])
			self.datasets["valid"].prefetch([int(k.split("_")[1]) for k in nns_keys_valid])
			for nns_key in nns_keys:
				nns_split, nns_id = nns_key.split("_")
				nns_id = int(nns_id)
				nns.append(self.datasets[nns_split][nns_id])
				if nns_map.get(nns_id, None) is None:
					nns_map[nns_id] = [idx.item()]
				else:
					nns_map[nns_id].append(idx.item())

		batch = self.datasets[split].collater(nns)
		# if this check fails it means the list nns was empty
		if "net_input" not in batch.keys():
			return None
		src_tokens = batch["net_input"]["src_tokens"]
		src_padding_mask = src_tokens.eq(self.src_dict.pad())
		trg_tokens = batch["target"]
		trg_padding_mask = trg_tokens.eq(self.trg_dict.pad())

		# nns_query_ids[i] is the source sentence id corresponding to
		# the i^th retrieved sentence in batch
		nns_query_ids = []
		for idx in batch["net_input"]["ids"]:
			nns_query_ids.append(nns_map[idx.item()].pop(0))

		return {
			"src_tokens": src_tokens.to(ids.device),
			"src_padding_mask": src_padding_mask.to(ids.device),
			"trg_tokens": trg_tokens.to(ids.device),
			"trg_padding_mask": trg_padding_mask.to(ids.device),
			"nns_query_ids": torch.tensor(nns_query_ids).to(ids.device),
			"nns_ids": batch["net_input"]["ids"]
		}

class CSTMInternalEncoder(TransformerDecoder):

	"""
	Used for encoding the retrieved source and retrieved target sentences.
	This class is basically identical to transformer.TransformerDecoder except
	there's no softmax at the end of the forward method. It's just an
	encoder with cross attention.
	"""

	def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True)
		# self.embed_out is a module in TransformerDecoder that is unused in the forward pass. For some reason,
		# unused modules break fairseq's distributed functionality when calculating gradients, so we have to
		# delete it.
		if hasattr(self, "embed_out"):
			del self.embed_out

	def forward(self, tokens, padding_mask, encoder_out):
		# embed positions
		positions = self.embed_positions(
			tokens
		) if self.embed_positions is not None else None

		# embed tokens and positions
		x = self.embed_scale * self.embed_tokens(tokens)

		if self.project_in_dim is not None:
			x = self.project_in_dim(x)

		if positions is not None:
			x += positions
		x = F.dropout(x, p=self.dropout, training=self.training)

		# (batch * n_retrieved) x seqlen x hidden -> seqlen x (batch * n_retrieved) x hidden
		x = x.transpose(0, 1)
		# decoder layers
		for layer in self.layers:
			x, attn = layer(
				x,
				encoder_out['encoder_out'],
				encoder_out['encoder_padding_mask'],
				None,
				self_attn_padding_mask=padding_mask
			)

		if self.normalize:
			x = self.layer_norm(x)

		if self.project_out_dim is not None:
			x = self.project_out_dim(x)

		return {
			"encoder_out": x, # (seqlen, batch * n_retrieved, hidden)
			"encoder_padding_mask": padding_mask # (batch * n_retrieved, seqlen)
		}

@register_model("cstm_transformer")
class CSTMTransformerModel(TransformerModel):

	@staticmethod
	def add_args(parser):
		TransformerModel.add_args(parser)
		parser.add_argument("--cstm-layers", metavar="N", type=int,
							 help='number of layers in each cstm transformer')
		parser.add_argument("--cstm-share-embeddings", action="store_true",
							 help="use the same embeddings in the cstm as in the \
							 main encoder and decoder")
		parser.add_argument('--cstm-n-retrieved', metavar='N', type=int, 
							 help='number of train sentences to return for each src \
							in the cstm')
		parser.add_argument("--cstm-nns-data", metavar='S', type=str,
							 help="path to the nearest neighbor data for cstm")

	@classmethod
	def build_model(cls, args, task):
		"""
		Build a new model instance. Very similar to transformer.Transformer.build_model.
		A lot of it is copied.
		"""

		base_cstm_architecture(args)

		if not hasattr(args, 'max_source_positions'):
			args.max_source_positions = 1024
		if not hasattr(args, 'max_target_positions'):
			args.max_target_positions = 1024

		src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

		def build_embedding(dictionary, embed_dim, path=None):
			num_embeddings = len(dictionary)
			padding_idx = dictionary.pad()
			emb = Embedding(num_embeddings, embed_dim, padding_idx)
			# if provided, load from preloaded dictionaries
			if path:
				embed_dict = utils.parse_embedding(path)
				utils.load_embedding(embed_dict, dictionary, emb)
			return emb

		if args.share_all_embeddings:
			if src_dict != tgt_dict:
				raise ValueError('--share-all-embeddings requires a joined dictionary')
			if args.encoder_embed_dim != args.decoder_embed_dim:
				raise ValueError(
					'--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
			if args.decoder_embed_path and (
					args.decoder_embed_path != args.encoder_embed_path):
				raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
			encoder_embed_tokens = build_embedding(
				src_dict, args.encoder_embed_dim, args.encoder_embed_path
			)
			decoder_embed_tokens = encoder_embed_tokens
			args.share_decoder_input_output_embed = True
		else:
			encoder_embed_tokens = build_embedding(
				src_dict, args.encoder_embed_dim, args.encoder_embed_path
			)
			decoder_embed_tokens = build_embedding(
				tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
			)

		if args.cstm_share_embeddings:
			cstm_src_embed_tokens = encoder_embed_tokens
			cstm_trg_embed_tokens = decoder_embed_tokens
		else:
			cstm_src_embed_tokens = build_embedding(
				src_dict, args.encoder_embed_dim, args.encoder_embed_path
			)
			cstm_trg_embed_tokens = build_embedding(
				tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
			)

		nns_data = torch.load(args.cstm_nns_data)

		cstm = CSTM(args, src_dict, cstm_src_embed_tokens, tgt_dict, \
					cstm_trg_embed_tokens, task.datasets, nns_data)

		encoder = CSTMTransformerEncoder(args, src_dict, encoder_embed_tokens, cstm)
		decoder = CSTMTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
		return CSTMTransformerModel(encoder, decoder)

	def forward(self, src_tokens, src_lengths, prev_output_tokens, ids, split):
		encoder_out = self.encoder(src_tokens, src_lengths, ids, split)
		decoder_out = self.decoder(prev_output_tokens, encoder_out)
		return decoder_out

@register_model_architecture("cstm_transformer", "cstm_transformer")
def base_cstm_architecture(args):
	base_architecture(args)
	args.cstm_share_embeddings = getattr(args, "cstm_share_embeddings", False)
	args.cstm_layers = getattr(args, "cstm_layers", 1)
	args.cstm_n_retrieved = getattr(args, "cstm_n_retrieved", 1)
	args.cstm_nns_data = getattr(args, "cstm_nns_data", "nns_ids_combined.pt")

