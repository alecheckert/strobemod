#!/usr/bin/env python
"""
radprojsim.py -- generate the matrices that allow numerical
projection of 3D radial displacement distributions into 2D

"""
import numpy as np 
import pandas as pd 
import dask 
from dask.diagnostics import ProgressBar 
from tqdm import tqdm 

def shell_project(bin_edges, R0, R1, delta_z, n_samples=10000000,
	batch_size=1000000):
	"""
	A particle starts at the Cartesian coordinate (0, 0, z), where
	z is a uniform random variable between -delta_z/2 and +delta_z/2.
	In one frame interval, it travels to a random point with a radial
	distance between R0 and R1 from its original position.

	The displacement is only observed if the final z position lies
	within the range (-delta_z/2, +delta_z/2). Further, only its XY
	position is observed; the z position is hidden (apart from 
	knowing that it lies in the aforementioned range).

	This function estimates the resulting distribution of observed 2D
	radial displacements in the XY plane. This is a vector over a set of
	radial bins. The sum of the vector is the probability that the final
	z position is between -delta_z/2 and +delta_z/2.

	args
	----
		bin_edges 		:	1D ndarray, the edges of the spatial bins to
							use for the final vector
		R0  			:	float, smallest radial distance traveled
		R1 				:	float, largest radial distance traveled 
		delta_z 		:	float, thickness of the observation slice
		n_samples		:	int, the number of samples to use in the Monte
							Carlo integration
		batch_size		:	int, the number of samples per iteration

	returns
	-------
		1D ndarray of shape (bin_edges.shape[0]-1), the fraction of
			XY radial displacements that fall into each bin

	"""
	hz = delta_z / 2.0 
	n_batches = int(np.ceil(n_samples / batch_size))
	n_samples = n_batches * batch_size 

	result = np.zeros(bin_edges.shape[0]-1, dtype=np.float64)

	for batch_idx in range(n_batches):

		# Choose a random 3D vector for each sample
		vectors = np.random.normal(size=(batch_size, 3))
		R = np.sqrt((vectors**2).sum(axis=1))
		vectors = (vectors.T * np.random.uniform(R0, R1, size=batch_size) / R).T

		# Choose a random starting position in z
		vectors[:,0] = vectors[:,0] + np.random.uniform(-hz, hz, size=batch_size)

		# Take only vectors that land inside the observation slice
		inside = np.abs(vectors[:,0]) <= hz 

		# Compute the XY radial displacements of these vectors
		r_xy = np.sqrt((vectors[inside,1:]**2).sum(axis=1))

		# Accumulate the histogram
		H, _ = np.histogram(r_xy, bins=bin_edges)
		result += H 

	# Normalize
	result = result / n_samples 

	return result 

def shell_proj_dist(bin_edges_3d, bin_edges_2d, delta_z, n_samples=100000000,
	batch_size=1000000, num_workers=4):
	"""
	Compute the result of shell_proj() for each of a set of radial distances
	in 3D, and return the resulting distribution.

	The output of this function can be used to renormalize any 3D radially
	symmetric displacement distribution into the distribution of 2D radial
	displacements expected for the HiLo geometry.

	args
	----
		bin_edges_3d:	1D ndarray, the set of radial displacement bins in 
						3D
		bin_edges_2d: 	1D ndarray, the set of bins for the 2D radial
						displacements
		delta_z		:	float, the thickness of the observation slice
		n_samples	:	int, the number of samples to use in the Monte Carlo
						integration of each bin
		batch_size 	:	int

	returns
	-------
		2D ndarray of shape (bin_edges_3d.shape[0]-1, bin_edges_2d.shape[0]-1),
			the distribution of 2D radial displacements for each 3D 
			radial displacement bin

	"""

	n_bins_3d = bin_edges_3d.shape[0] - 1
	n_bins_2d = bin_edges_2d.shape[0] - 1

	if num_workers == 1:
		results = np.zeros((n_bins_3d, n_bins_2d), dtype=np.float64)
		for r0_idx in tqdm(range(n_bins_3d)):
			r0 = bin_edges_3d[r0_idx]
			r1 = bin_edges_3d[r0_idx+1]
			results[r0_idx, :] = shell_project(bin_edges_2d, r0, r1, delta_z,
				n_samples=n_samples, batch_size=batch_size)

	else:
		@dask.delayed 
		def run_shell_proj(r0_idx):
			r0 = bin_edges_3d[r0_idx]
			r1 = bin_edges_3d[r0_idx+1]
			result = shell_project(bin_edges_2d, r0, r1, delta_z,
				n_samples=n_samples, batch_size=batch_size)
			return result 

		results = [run_shell_proj(i) for i in range(n_bins_3d)]
		with ProgressBar():
			results = dask.compute(*results, scheduler="processes", num_workers=num_workers)
		results = np.asarray(results)

	return results

# Sample usage
if __name__ == '__main__':
	bin_edges_3d = np.linspace(0.0, 5.0, 5001)
	bin_edges_2d = np.linspace(0.0, 5.0, 5001)
	result = shell_proj_dist(
		bin_edges_3d,
		bin_edges_2d,
		0.7,
		n_samples=100000000,
		num_workers=8
	)
	result = pd.DataFrame(result, columns=bin_edges_2d[:-1])
	result["R_3d"] = bin_edges_3d[:-1]
	result.to_csv("radial_proj_dz-0.7_1e8samples.csv", index=False)
	print(result)
	print(result.columns)




