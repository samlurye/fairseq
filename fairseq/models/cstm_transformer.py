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

class CSTMTransformerDecoderLayer(TransformerDecoderLayer):

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
			cs, attn = self.encoder_attn(
				query=x,
				key=encoder_out["encoder"],
				value=encoder_out["encoder"],
				key_padding_mask=encoder_padding_mask["encoder"],
				incremental_state=incremental_state,
				static_kv=True,
				need_weights=(not self.training and self.need_attn),
			)
			cm, _ = self.cstm_attn(
				query=x,
				key=encoder_out["cstm"],
				value=encoder_out["cstm"],
				key_padding_mask=encoder_padding_mask["cstm"],
				incremental_state=incremental_state,
				static_kv=True,
				need_weights=(not self.training and self.need_attn),
			)
			g = (self.W_gs(cs) + self.W_gm(cm)).sigmoid()
			x = g * x + (1 - g) * x
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

	def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn, left_pad, final_norm)
		self.layers = nn.ModuleList([])
		self.layers.extend([
			CSTMTransformerDecoderLayer(args, no_encoder_attn)
			for _ in range(args.decoder_layers)
		])

class CSTMTransformerEncoder(TransformerEncoder):

	def __init__(self, args, src_dict, encoder_embed_tokens, cstm):
		super().__init__(args, src_dict, encoder_embed_tokens)
		self.cstm = cstm

	def forward(self, src_tokens, src_lengths):
		encoder_out = super().forward(src_tokens, src_lengths)
		# do this nonsense so that the signature of CSTMTransformerDecoderLayer.forward
		# is the same as the signature of TransformerDecoderLayer.forward,
		# which means we can just use TransformerDecoder.forward for 
		# CSTMTransformerDecoder.forward
		cstm_out, cstm_padding_mask = self.cstm(
			src_tokens, 
			encoder_out
		)
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

	def __init__(self, args, src_dict, src_embed_tokens, trg_dict, trg_embed_tokens):
		super().__init__()

		new_args = argparse.Namespace(**vars(args))
		new_args.decoder_layers = args.cstm_layers

		self.retrieved_src_encoder = CSTMInternalEncoder(new_args, src_dict, src_embed_tokens)
		self.retrieved_trg_encoder = CSTMInternalEncoder(new_args, trg_dict, trg_embed_tokens)

		self.n_retrieved = args.cstm_n_retrieved

	def forward(self, src_tokens, encoder_out):
		n_retrieved = self.n_retrieved
		retrieved = self.retrieve(src_tokens, encoder_out["encoder_padding_mask"], n_retrieved)
		retrieved_src_encoding = self.retrieved_src_encoder(
			retrieved["src_tokens"], 
			retrieved["src_padding_mask"],
			encoder_out
		)
		retrieved_trg_encoding = self.retrieved_trg_encoder(
			retrieved["trg_tokens"],
			retrieved["trg_padding_mask"],
			retrieved_src_encoding
		)
		batch = src_tokens.size(0)
		seqlen = retrieved["trg_tokens"].size(1)
		tmp = retrieved_trg_encoding["encoder_out"]
		tmp = tmp.transpose(0, 1) # (batch * n_retrieved) x seqlen x hidden
		tmp = tmp.reshape(batch, n_retrieved * seqlen, -1) # batch x (n_retrieved * seqlen) x hidden
		tmp = tmp.transpose(0, 1) # (n_retrieved * seqlen) x batch x hidden
		retrieved_trg_encoding["encoder_out"] = tmp
		if retrieved_trg_encoding["encoder_padding_mask"] is not None:
			tmp = retrieved_trg_encoding["encoder_padding_mask"]
			tmp = tmp.reshape(batch, n_retrieved * seqlen)
			retrieved_trg_encoding["encoder_padding_mask"] = tmp
		return retrieved_trg_encoding["encoder_out"], retrieved_trg_encoding["encoder_padding_mask"]

	def retrieve(self, src_tokens, src_padding_mask, n_retrieved):
		# TODO: actually implement this with nearest neighbors search
		# returns:
		# *_tokens: LongTensor of shape (batch * n_retrieved, seqlen)
		# *_padding_mask is a ByteTensor of shape (batch * n_retrieved, seqlen) indicating
		# the locations of padding elements
		#
		# In order to interface with everything else, for 0 <= i < batch and
		# 0 <= j < n_retrieved, row n_retrieved * i + j of each returned tensor
		# needs to be the values associated with the j^th retrieved sentence
		# for source sentence i. That is, as opposed to indexing by batch * j + i.
		return {
			"src_tokens": torch.ones_like(src_tokens.repeat(n_retrieved, 1)),
			"src_padding_mask": torch.zeros_like(src_padding_mask.repeat(n_retrieved, 1)) \
								 if src_padding_mask is not None else None,
			"trg_tokens": torch.ones_like(src_tokens.repeat(n_retrieved, 1)),
			"trg_padding_mask": torch.zeros_like(src_padding_mask.repeat(n_retrieved, 1)) \
								 if src_padding_mask is not None else None
		}

class CSTMInternalEncoder(TransformerDecoder):

	def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True):
		super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False, left_pad=False, final_norm=True)
		# self.embed_out is a module in TransformerDecoder that is unused in the forward pass. For some reason,
		# unused modules break fairseq's distributed functionality when calculating gradients, so we have to
		# delete it.
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

	@classmethod
	def build_model(cls, args, task):
		"""Build a new model instance."""

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

		cstm = CSTM(args, src_dict, cstm_src_embed_tokens, tgt_dict, cstm_trg_embed_tokens)

		encoder = CSTMTransformerEncoder(args, src_dict, encoder_embed_tokens, cstm)
		decoder = CSTMTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
		return CSTMTransformerModel(encoder, decoder)

@register_model_architecture("cstm_transformer", "cstm_transformer")
def base_cstm_architecture(args):
	base_architecture(args)
	args.cstm_share_embeddings = getattr(args, "cstm_share_embeddings", True)
	args.cstm_layers = getattr(args, "cstm_layers", 1)
	args.cstm_n_retrieved = getattr(args, "cstm_n_retrieved", 1)

