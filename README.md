# Caizi 才子

<img src="cover.png" width="40%">


## Description
This project aims to create a digital talent (才子) who can write novels, paint, and compose music. Under the hood, the lastest AI  will be used, such as GPT-like  family, diffusion models  and other generative technologies. To a large sense, the project is of educational purpose, to learn the latest AI technologies and to have fun.

For now, as the first step, the project will focus on the text generation part.  

There have been lots of GPT-2 implementations on web, but in this project, the focus is on the educational aspect, to learn the model from scratch, to understand the model, to implement the model, to train the model, to fine-tune the model, to evaluate the model, to deploy the model, and to use the model.

The project also pays attention to the following aspects:

1. Software engineering best practices:
  -  Object-oriented programming for better code organization, readability, and maintainability 
  -  Test-driven development for better code quality
  -  Continuous build and integration for better code quality

2. Distributed computing:
  - Hetoerogeneous computing with consitent framework
  - Easy to scale up and down
  - Easy to develop and deploy 
   


## Table of Contents

- [Caizi 才子](#caizi-才子)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install python dependencies](#install-python-dependencies)
  - [Usage](#usage)
  - [Tools](#tools)
  - [Contributing](#contributing)
  - [License](#license)
  - [Reference](#reference)

## Installation

### Prerequisites

Docker environment is required to run the project in order to isolates the project's dependencies and configurations. Please refer to the [official documentation](https://docs.docker.com/get-docker/) for installation instructions.

More specifically, since vscode is used as the development environment, the Docker DevContainer is used to ensure that the development environment is consistent across different machines. Please refer to the [official documentation](https://code.visualstudio.com/docs/remote/containers) for installation instructions.

After installing Docker and vscode, clone the repository and open the project in vscode. The `.devcontainer` folder is present in the main `CaiZi` directory. VS Code should automatically detect it and ask whether you would like to open the project in a devcontainer. If it doesn't, simply press `Ctrl + Shift + P` to open the command palette and start typing `dev containers` to see a list of all DevContainer-specific options.


Now select **Reopen in Container**. Docker will now begin the process of building the Docker image specified in the .devcontainer configuration if it hasn't been built before, or pull the image if it's available from a registry.

The entire process is automated and might take a few minutes, depending on your system and internet speed. Optionally click on "Starting Dev Container (show log)" in the lower right corner of VS Code to see the current built progress.

Once completed, VS Code will automatically connect to the container and reopen the project within the newly created Docker development environment. You will be able to write, execute, and debug code as if it were running on your local machine, but with the added benefits of Docker's isolation and consistency.

> [!WARNING]
> It is a must to run with GPU support, as the project will use the latest AI models, which are computationally expensive.

> [!WARNING]
> If you are encountering an error during the build process, this is likely because your machine does not support NVIDIA container toolkit because your machine doesn't have a compatible GPU. In this case, edit the `devcontainer.json` file to remove the `"runArgs": ["--runtime=nvidia", "--gpus=all"],` line and run the "Reopen Dev Container" procedure again.


### Install python dependencies
You don't have to install any python dependencies, as they are already installed in the Docker container. Please note the 
dependencies are listed in the `requirements.in` file which is used to generate the `requirements.txt` file with better version control.


## Usage

Guidelines on how to use the project.


## Tools


## Contributing

[Bin.Li](mailto:ornot2008@yahoo.com) is the sole contributor to this project. However, if you would like to contribute,dont hesitate to contact me.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 


## Reference
