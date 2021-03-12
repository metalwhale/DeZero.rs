# `DeZero` infrastructure

## How to run
- Create `.env` file and specify variables:
  ```bash
  $ cp .env.example .env
  ```
- Startup and get into the container:
  ```bash
  $ docker-compose up -d
  $ docker-compose exec dezero bash
  ```
- Install VSCode extensions inside the container:
  - `tamasfe.even-better-toml`: [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
  - `matklad.rust-analyzer`: [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=matklad.rust-analyzer)
  - `vadimcn.vscode-lldb`: [CodeLLDB](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)
