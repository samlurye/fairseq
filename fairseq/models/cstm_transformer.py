import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq.modules import MultiheadAttention
from . import register_model, register_model_architecture
from .transformer import *
from fairseq import utils

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

	def __init__(self, args, src_dict, encoder_embed_tokens):
		super().__init__(args, src_dict, encoder_embed_tokens)
		self.cstm = CSTM(args)

	def forward(self, src_tokens, src_lengths):
		encoder_out = super().forward(src_tokens, src_lengths)
		# do this nonsense so that the signature of CSTMTransformerDecoderLayer.forward
		# is the same as the signature of TransformerDecoderLayer.forward,
		# which means we can just use TransformerDecoder.forward for 
		# CSTMTransformerDecoder.forward
		cstm_out, cstm_padding_mask = self.cstm.retrieve(
			encoder_out["encoder_out"], 
			encoder_out["encoder_padding_mask"]
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

class CSTM:

	def __init__(self, args):
		pass

	def retrieve(self, encoder_out, encoder_padding_mask):
		return encoder_out, encoder_padding_mask

@register_model("cstm_transformer")
class CSTMTransformerModel(TransformerModel):

	@staticmethod
	def add_args(parser):
		TransformerModel.add_args(parser)
		# TODO: add args to the parser for CSTM class to use

	@classmethod
	def build_model(cls, args, task):
		"""Build a new model instance."""

		# make sure all arguments are present in older models
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

		encoder = CSTMTransformerEncoder(args, src_dict, encoder_embed_tokens)
		decoder = CSTMTransformerDecoder(args, tgt_dict, decoder_embed_tokens)
		return CSTMTransformerModel(encoder, decoder)

@register_model_architecture("cstm_transformer", "cstm_transformer")
def base_cstm_architecture(args):
	base_architecture(args)