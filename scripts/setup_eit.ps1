$ErrorActionPreference = "Stop"

. "$PSScriptRoot\assert_eit.ps1"
Assert-EitEnvironment

python -m pip install -e ".[test]"
python -c "import torch, torch_geometric, gensim, transformers; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('torch_geometric=', torch_geometric.__version__); print('gensim=', gensim.__version__); print('transformers=', transformers.__version__)"
