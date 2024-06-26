# Unveiling the Scheduling Mechanisms of NVIDIA RTX 4090 GPU: Insights for Enhancing Covert Attacks

This is the repository for the final project titled "Unveiling the Scheduling Mechanisms of NVIDIA RTX 4090 GPU," part of the COMS 6424 Hardware Security course at Columbia University, Spring 2024. This project investigates the scheduling policies and block-to-SM allocation strategies of the NVIDIA RTX 4090 GPU. Through comprehensive experiments, we analyze factors such as kernel queueing, resource availability, and stream priorities within GPU scheduling.

## Repository Structure
- **data/**: Raw experimental json data.
- **include/**: Header files for project source code.
- **meeting_slide/**: Slides and presentations for project meetings.
- **profiling_results/**: Outputs from profiling tools and analysis scripts.
- **scripts/**: Scripts to build and run the experiments.
- **src/**: Source files for the project.
- **README.md**: This file, containing project information and setup instructions.
- **Unveiling_the_Scheduling_Mechanisms_of_NVIDIA_RTX4090.pdf**: The project report detailing our findings and methodologies.

## Setup and Execution
### Prerequisites:
- **Operating System**: Windows 10
- **CUDA Version**: 12.2
- **Compiler**: Microsoft Visual Studio (MSVC)

To set up and run the experiments:
1. Navigate to the `scripts/` directory.
2. Execute the build scripts to generate executable files.
3. Run the generated `.exe` files from the `build/` directory.

To run the worst-case scheduling experiments:
1. Navigate to the `WorstSchedulingExps/` directory.
2. Run `make tests` to build the programs.
3. Run the generated executable files.

**Note**: 
- Due to NVCC's inability to recognize MSVC paths set in our system environment variables, paths to MSVC are hardcoded in the scripts. Please update these paths if your setup differs.

- The TPC masking function is specific to the linux version of the CUDA kernel launch function, so worst-case scheduling experiments should only run on linux.

## Paper
For a detailed discussion of the experiments and findings, please refer to the included PDF document.

Thank you for exploring our project on GPU scheduling insights!