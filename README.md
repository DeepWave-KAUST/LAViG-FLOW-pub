# LAViG-FLOW: Latent Autoregressive Video Generation for Fluid Flow Simulations

![Image](https://github.com/user-attachments/assets/071bd192-9586-4f16-bbde-3b02c7a34aeb)

Reproducible material for **LAViG-FLOW: Latent Autoregressive Video Generation for Fluid Flow Simulations - De Pellegrini V., Alkhalifah T.**

## :bar_chart: Project structure
This repository is organized as follows:

* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **gas_saturation_vqvae**: set of python scripts to build the VQ-VAE latent space for the CO₂ Gas Saturation field;
* :open_file_folder: **pressure_buildup_vae**: set of python scripts to build the VAE latent space for the Pressure Build-Up field;
* :open_file_folder: **gas_saturation_pressure_buildup_ditv**: set of python scripts to train latent autoregressive video diffusion transformer (DiTV);
* :open_file_folder: **stylegan-v-main**: StyleGAN-V metric utilities to evaluate video-generation quality (FVD/LPIPS, etc.). Source: https://github.com/universome/stylegan-v;
* :open_file_folder: **dockers**: Dockerfile for Docker users who need to build the docker image.


## :space_invader: :robot: Getting started 
> [!IMPORTANT]
>To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.
>
>Simply run:
>```
>./install_env.sh
>```
>It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 
>
>Remember to always activate the environment by typing:
>```
>conda activate lavig-flow
>```

## 🐳 Docker support
Prebuilt images live on Docker Hub: https://hub.docker.com/r/vittodepe98/lavig-flow.

```bash
docker pull vittodepe98/lavig-flow:latest
```

You can execute the `docker pull` command from any directory, but running it while your shell is already inside the cloned `LAViG-FLOW/` folder keeps you in the right context for the follow-up `docker run` step.

The published Docker Hub image is generated from `LAViG-FLOW/dockers/lavig_flow.Dockerfile`. Use it whenever you want to work on the repository without installing the environment locally—the container behaves like a ready-to-go virtual environment with CUDA, PyTorch, and all dependencies baked in.

From a terminal opened in your cloned `LAViG-FLOW/` folder, mount the repo checkout and drop into the container (Linux/macOS shells or Windows PowerShell):

```bash
docker run --rm -it --gpus all --name lavig-flow-local -v ${PWD}:/mycode -w /mycode vittodepe98/lavig-flow:latest bash
```

This command assumes you already cloned the repository locally (`git clone ... LAViG-FLOW`) and pulled the Docker image. It then:

- mounts your current working tree into `/mycode` inside the container (`-v ${PWD}:/mycode`) and sets that as the working directory (`-w /mycode`), so any file edits you do on the host (e.g., in VS Code) show up instantly inside the container and vice versa;
- enables GPU passthrough with `--gpus all`;
- starts an interactive shell (`-it`) that is automatically cleaned up afterwards (`--rm`) and is easy to find/attach to (`--name lavig-flow-local`).

Once the shell comes up you can run scripts normally (e.g., `python gas_saturation_vqvae/train.py ...`) and everything executes with the container’s pre-installed toolchain. Editors like VS Code can also “Attach to Running Container” (Remote-Containers extension) to provide a virtual dev environment directly inside `lavig-flow-local`.

> [!NOTE]
>**Disclaimer:** All experiments have been run on KAUST’s IBEX cluster (dual Intel® Xeon® compute nodes paired with NVIDIA Tesla V100 GPUs). Different environment 
>configurations may be required for different combinations of workstation and GPU.

## Acknowledgements
We deeply appreciate the upstream projects and datasets that make LAViG-FLOW possible across its many submodules. You will find references embedded throughout the individual scripts (pointing to original papers, datasets, and codebases); we extend our thanks to every one of those researchers and maintainers for their foundational contributions, and to the KAUST IBEX support team for providing the NVIDIA V100 infrastructure used in our experiments.

## :alien: :flying_saucer: :cow2: Cite us 

```bibtex
@misc{depellegrini2026lavigflowlatentautoregressivevideo,
      title={LAViG-FLOW: Latent Autoregressive Video Generation for Fluid Flow Simulations}, 
      author={Vittoria De Pellegrini and Tariq Alkhalifah},
      year={2026},
      eprint={2601.13190},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.13190}, 
}
