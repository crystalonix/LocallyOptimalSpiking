# LocallyOptimalSpiking

This project implements a framework for encoding continuous time signals using spiking neurons.

## Project Overview

1. **Module Descriptions**
    - Brief descriptions of the various modules in the project, highlighting their functionalities.

2. **Running the Project**
    - Instructions on how to run the project in different modes.

4. **Licensing and Authorship**
    - Information about the licensing terms and authorship.

## Module Descriptions

The following diagram shows the dependencies between different modules in the project:
![Dependency Diagram](images/dependency_diagram.png)
Key modules are briefly described below.
- **direct_reconstruction_driver:** Main module that provides the interface for running the large scale experiments over large number of audio snippets by systematically varying the model parameter and generates the statistics of the whole experiment. The module overrides the default parameter set in the configuration file with a range of parameter values set inline. 
- **reconstruction_driver:** Module that provides interface for processing a single audio snippet by its index from the repository of audio snippets, inputted as text files. 
- **reconstruction_manager:** Module that provides an interface for orchestrating the whole processing of coding and reconstruction for a given input signal with a given set of kernels. The module achieves this task by exposing a bunch apis that centralize the process of signal kernel convolutions, invokes the spike generator and finally reconstructs the signal iteratively or in batch-mode (depending on the mode of execution), generate statistics and so on.
- **gammatone_calculator:** Computes gammatone kernels for 1D audio signal processing. Kernels correspond to frequencies evenly spaced on the ERB scale, with 5,000 center frequencies listed in 'frequencies_5000.txt'. 
This module supports both compressed (using coefficients of shifted B-splines) and uncompressed kernel calculations. The compressed form can speed up inner product computations, especially when learning coefficients for experimentation.
- **kernel_manager:** Sits above the concrete kernel generator modules (e.g., gammatone_calculator). It initializes the framework with the required number of kernels and provides an interface for functionalities like computing inner products between kernels. See inline documentation for details.
- **spike_generator:** Concrete implementation module for generating spikes, given an input signal and the set of kernels, using a simple convolve and threshold scheme. 
- **configuration:** Configuration file that stores the initial values of parameters used in the project.
- **common_utils:** Module that provides common utility methods for the projects supporting operations, e.g. matrix operations, interpolation etc.
- **signal_utils:** Module that provides utility methods supporting several operations on the input/ output or convolution signals. Operations include calculating norm, upsample, calculating convolution, etc.
- **plot_utils:** Utility module that provides several plotting functionalities used for visualization purposes.
- **convolutional_sparse_coding:** This module leverages the state of the art convolutional sparse coding techniques from the sporco package runs large scale experiments on repository of audio signal with a given set kernels similar to direct_reconstruction_driver. In this case, the parameters of cbpdn are systematically varied to achieved reconstructions are different spike rates and the values of the parameters can be found inline.
- **iterative_spike_generator:** This is an experimental module, independent of the rest of the framework whereupon we seek to generate spike by setting some local optimality criteria by setting gradient wrt to the reconstruction error to zero. This experimental module, for a given input generates the spike iteratively as described and also reconstructs the signal iteratively. 
- **slim_reconstruction_driver:** This is another experimental module that provides an interface for coding and reconstruction of input signals by invoking the experimental **iterative_spike_generator**. Parameters of the experiment can be found inline.
## Running the Project

To use this project, follow these steps:

1. **Library Dependency:**
   The following libraries are required for running the project: **numpy** (1.26.3), **pandas** (2.1.4), **seaborn** (0.13.1), **tensorflow** (2.15.0), **torch** (1.9.0).

2. **Input:**
   The project mainly supports input audio signals in '.txt' format. The **wav_file_handler** module extracts a float array from a single-channel audio file in **wav** format and vice versa. The project experiments are run against the Freesound audio repository, with audio snippets converted from **.wav** to **.txt** format under the **audio_txt** folder. The original project was tested against thousands of **.txt** audio snippets, but for brevity, only a subsample of about a hundred audio **.txt** files is included in the GitHub repository.

3. **Running Spike-Based Encoding on Audio Signals:**
   The main module responsible for spike-based encoding of audio snippets in '.txt' format is the **direct_reconstruction_driver**. Input audio **.txt** files are read through integer indexes, i.e., an input audio text file named '#i.txt' is indexed as integer #i in the driver. The driver has been run in different modes of execution for various experiments, each with a predefined set of values for input parameters or model parameters found inline in the code. All output statistics are generated in the file specified by the variable 'stats_csv_file'. Each important mode of execution is briefly described below:
   
	- **large_experiment_mode:** This mode is used for large-scale experiments testing the spike coding framework over a large set of audio signals. Parameters include the array 'sample_numbers' indicating the range of input audio **.txt** files for encoding, snippet length (approximately 2.5s, indicated by 'sample_len'), and number of kernels (typically up to 100, indicated by 'number_of_kernel'). Other model parameters, such as 'ahp' constants, window size, etc., can be found inline.

	- **csc_experiment_mode:** Similar to 'large_experiment_mode', but with slightly different parameter values to obtain results of spike coding to compare against convolutional sparse coding techniques. In this mode, sample lengths vary from 0.5s to 2.5s (indicated by 'sample_lens_array'), fewer kernels are used (number_of_kernel set to 10), and experiments are run on a smaller set of signals. Other parameters values can be found inline.
   
	- **demo_mode:** Used for executing a single file, mainly for generating demo plots. In this mode the variable 'sample_number' indicates the index of the audio **.txt** file chosen from the 'audio_txt_' folder.

   To run an experiment on new audio files:
   	i. Generate the audio **.txt** files.
   	ii. Place them in the **audio_txt_** folder, naming each with a unique integer ID.
    iii. Choose the appropriate mode and set the `sample_number` (for demo_mode) or `sample_numbers` array (for the other modes). Changing preset model parameters is at the user's discretion.

4. **Running Convolutional Sparse Coding:**
   The module named **convolutional_sparse_coding** is used for running convolutional sparse coding (cbpdn) on audio snippets leveraging the state-of-the-art **sporco** library. Similar to the spike-based encoding module, it runs on audio **.txt** files indexed by integers. The variable 'sample_numbers' indicates the range of indexes for the audio signals used as input. Typically, 10 kernels are used (indicated by 'number_of_kernel'), and the length of the audio snippets is varied from 0.5s to 2.5s for performance evaluation. The driver specifies a range of values for the regularization parameter 'lambda' used by the 'cbpdn' procedure (indicated by the array 'lambdas'). Varying the 'lambda' values systematically enables comparison of reconstructions at different levels of sparsity. To run convolutional sparse coding on a set of audio snippets:
    i. Place the input audio **.txt** files in the appropriate location.
    ii. Specify the integer indexes of those input files in the variable 'sample_numbers'. Other parameters can be found inline and adjusted as needed.

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).

![CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)

**Important Note**: This project is also subject to a provisional patent. The Creative Commons license applies to the documentation and code provided herein, but does not grant any rights to the patented invention.

## Authors

- Anik Chattopadhyay (https://github.com/crystalonix) 
- Arunava Banerjee
