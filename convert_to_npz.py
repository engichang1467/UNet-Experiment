import h5py
import numpy as np
import argparse
import glob


def main():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('-f', '--file_pattern', required=True, type=str,
	                    help='File pattern of h5 files.')
	parser.add_argument('-o', '--outfile', type=str, default='snapshots.npz',
	                    help='Ouput file name.')

	args = parser.parse_args()
	files = sorted(glob.glob(args.file_pattern))

	scales = ['write_number', 'sim_time']

	# convert velocity into horizontal and vertical velocity
	variables = ['buoyancy', 'pressure', 'horizontal_velocity', 'vertical_velocity', 'vorticity']

	var_dict = {}
	
	for key in (variables + scales):
		var_dict[key] = []

	for file in files:
		fh = h5py.File(file, 'r')
		for s in scales:
			var_dict[s].append(np.array(fh['scales'][s]))
		for v in variables:

			# Convert velocity (81, 2, 128, 64) --> horizontal_velocity (81, 128, 64), vertical_velocity (81, 128, 64)
			if v == 'horizontal_velocity':
				horizontal_velocity = np.array(fh['tasks']['velocity'][:, 0, :, :])
				var_dict[v].append(horizontal_velocity)
			
			elif v == 'vertical_velocity':
				vertical_velocity = np.array(fh['tasks']['velocity'][:, 1, :, :])
				var_dict[v].append(vertical_velocity)
			
			else:
				var_dict[v].append(np.array(fh['tasks'][v]))

	for key in var_dict.keys():
		var_dict[key] = np.concatenate(var_dict[key], axis=0)

	# sort based on write_number
	sort_idx = np.argsort(var_dict['write_number'])
	for key in var_dict.keys():
		var_dict[key] = var_dict[key][sort_idx].astype(np.float32)
		
	np.savez(args.outfile, **var_dict)

if __name__ == '__main__':
	main()
