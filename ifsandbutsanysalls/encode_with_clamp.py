#!/usr/bin/env python3
"""
Example command:

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas
  -o profiles/profile-optional-clamp
  --force-overwrite true
  python3 encode_with_clamp.py
"""
import click
import torch
from torch import nn
from tqdm import tqdm


class Model(nn.Module):
	def __init__(self, max_len: int, d_model: int, n_head: int, n_layer: int, always_clamp: bool = False):
		super().__init__()
		self.always_clamp = always_clamp
		self.pos_embedding = nn.Embedding(max_len, d_model)
		self.layers = nn.ModuleList([
			nn.TransformerEncoderLayer(
				d_model, n_head, dim_feedforward=4*d_model, batch_first=True
			) for _ in range(n_layer)
		])

	def forward(self, x: torch.tensor, x_len: torch.tensor) -> torch.tensor:
		idxes = torch.arange(x.size(1), device=x.device)
		torch.cuda.nvtx.range_push("pos_embed")
		x = x + self.pos_embedding(idxes)
		torch.cuda.nvtx.range_pop()
		mask = idxes.unsqueeze(0) >= x_len.unsqueeze(1)
		for layer in self.layers:
			torch.cuda.nvtx.range_push("encoder-layer")
			x = layer(x, src_key_padding_mask=mask)
			torch.cuda.nvtx.range_pop()
			if self.always_clamp or (x.abs() > 100).any():
				torch.cuda.nvtx.range_push("clamp")
				x = torch.clamp(x, -100, 100)
				torch.cuda.nvtx.range_pop()
		return x


@click.command()
@click.option("--batch-size", type=int, default=128)
@click.option("--num-batches", type=int, default=5)
@click.option("--max-len", type=int, default=1024)
@click.option("--d-model", type=int, default=512)
@click.option("--n-head", type=int, default=8)
@click.option("--n-layer", type=int, default=6)
@click.option("--always-clamp", is_flag=True)
def main(batch_size, num_batches, max_len, d_model, n_head, n_layer, always_clamp):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model(max_len, d_model, n_head, n_layer, always_clamp).to(device)
	model = model.to(device)
	model.eval()

	for _ in tqdm(range(num_batches)):
		x = torch.rand(batch_size, max_len, d_model, device=device)
		x_len = torch.randint(1, max_len, (batch_size,), device=device)
		with torch.no_grad():
			y = model(x, x_len)


if __name__ == "__main__":
	main()