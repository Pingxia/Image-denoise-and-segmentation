# Requirements
- python3
- scipy
- matplotlib

# Folder structure
1. a1/: input images for a1
2. a2/: input images for a2
3. code/: code implementation for each task
	* gibbs_sampler.py: task 1, image denoise
	* variational_sampler.py: task 2, image denoise
	* em.py: task 3, image segmentation
4. output/: output images produced by code

Please keep this folder structure so that code can run properly.

# a1: Gibbs sampler
- inside code/ folder, run gibbs_sampler.py
- Output images will be generated in output/a1/ folder, named as [image number]_denoise_gibbs.png

# a1: Variational sampler
- inside code/ folder, run variational_sampler.py
- Output images will be generated in output/a1/ folder, named as [image number]_denoise_variational.png

# a2: EM algorithm
- inside code/ folder, run em.py
- Output images will be generated in output/a2/ folder
