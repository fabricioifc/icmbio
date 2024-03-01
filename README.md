# Redes Neurais Convolucionais para Segmentação Semântica de Imagens de Satélite do Google Earth

Este projeto tem como objetivo a segmentação semântica de imagens de satélite do Google Earth. Para isso, foi inicialmente criado um dataset com 256 imagens de satélite de 2048x2048 pixels (214 para treinamento e 42 para teste), contendo 8 classes: urbano, mata, sombra, regeneração, agricultura, rocha, solo e água. O dataset obtido foi anotado majoritariamente manualmente, utilizando a ferramenta [CVAT](https://github.com/opencv/cvat).

## Procedimento para execução do código

Esse procedimento foi testado no sistema operacional `Windows 10 64 bits`. Para outros sistemas operacionais, pode ser necessário realizar algumas adaptações. Siga os passos abaixo para executar o código:

## 0. Download do dataset

O dataset pode ser baixado [aqui](https://drive.google.com/file/d/1rmxywtjX18U5acpbUo4q8kS20anHIhKt/view). Após o download, extraia o arquivo `dataset.zip` para uma pasta de sua preferência. O resultado será uma pasta chamada `all` contendo as imagens de treinamento e teste. Essa pasta deve ser informada no parâmetro `root_dir` do arquivo `main.py`.

## 1. Instalação das ferramentas
 - [Python](https://www.python.org/downloads/) (versão 3.10.11)
 - [pip](https://pip.pypa.io/en/stable/installation/) (versão 23.3.1)
 - [Git](https://git-scm.com/downloads)
 - [vscode](https://code.visualstudio.com/download)

> **Observação:** É recomendado a instalação do Python e do Git com a opção de adicionar ao PATH.

Verifique se as ferramentas foram instaladas corretamente executando os seguintes comandos no terminal:

```bash
python --version
pip --version
git --version
```

## 2. Clonar o repositório

```bash
git clone https://github.com/fabricioifc/icmbio.git
git checkout alan
```

## 3. Instalação da biblioteca Virtualenv

```bash
pip install virtualenv
```

## 4. Criação do ambiente virtual

```bash
cd icmbio
virtualenv .venv
```

## 5. Ativação do ambiente virtual

```bash
.venv\Scripts\activate # Windows
source .venv/bin/activate # Linux
```

## 6. Instalação das bibliotecas

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall
pip install -r requirements.txt
```

## 7. Execução do código

```bash
python main.py
```

## Parâmetros de execução (default)

```python
params = {
    'results_folder': 'tmp\\20231211_cross_entropy_100_epoch_segnet_focalloss', # Pasta onde serão salvos os resultados
    'root_dir': 'D:\\datasets\\ICMBIO_NOVO\\all', # Diretório raiz dos dados
    'window_size': (256, 256), # Tamanho das imagens de entrada da rede
    'cache': True,
    'bs': 8, # Batch size
    'n_classes': 8, # Número de classes
    'classes': ["Urbano", "Mata", "Sombra", "Regeneracao", "Agricultura", "Rocha", "Solo", "Agua"], # Nome das classes
    'cpu': None, # CPU ou GPU. Se None, será usado GPU. Não vai funcionar com CPU
    'device': 'cuda', # GPU
    'precision' : 'full', # Precisão dos cálculos. 'full' ou 'half'. 'full' é mais preciso, mas mais lento. 'half' é mais rápido, mas menos preciso. Default: 'full'
    'optimizer_params': {
        'optimizer': 'SGD',
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005
    },
    'lrs_params': {
        'type': 'multi',
        'milestones': [25, 35, 45],
        'gamma': 0.1
    },
    'weights': '', # Peso de cada classe para a loss. Será calculado automaticamente em seguida
    'maximum_epochs': 100, # Número de épocas de treinaento
    'save_epoch': 10, # Salvar o modelo a cada n épocas para evitar perder o treinamento caso ocorra algum erro ou queda de energia
    'print_each': 500, # Print each n iterations (apenas para acompanhar visualmente o treinamento)
    'loss': {
        'name': LossFN.CROSS_ENTROPY, # Escolha entre 'CROSS_ENTROPY' ou 'FOCAL_LOSS'
        'params': {
            'weights': 'calculate', # Escolha entre 'equal' ou 'calculate'. Se 'equal', os pesos serão iguais. Se 'calculate', os pesos serão calculados pelo arquivo `extra\weights_calculator.py`
            'alpha': 0.5, # Somente para FOCAL_LOSS. Informe um valor float. Default: 0.5
            'gamma': 2.0, # Somente para FOCAL_LOSS. Informe um valor float. Default: 2.0
        }
    },
    'model': {
        'name': ModelChooser.UNET, # Escolha entre 'segnet_modificada' ou 'unet
    },
}
```

## Imagens de entrada

> Para definir as imagens que serão usadas no treinamento e teste, basta alterar o arquivo train_images.txt e test_images.txt, respectivamente. Cada linha do arquivo deve conter o nome da imagem, com a extensão. Por exemplo, se a imagem for `D:\datasets\ICMBIO_NOVO\all\001.tif`, então a linha deve ser `001.tif`.

## Observações

- O código foi testado com a GPU NVIDIA GeForce RTX 3060 12GB.
- O código foi testado com o sistema operacional `Windows 10 64 bits`.
- O código foi testado com o Python 3.10.11.

## Equipe de trabalho

- [Andre de Souza Brito](https://www.linkedin.com/in/andr%C3%A9-de-souza-brito-4b0a62a2/)
- [Fabricio Bizotto (UTFPR/IFC)](https://www.linkedin.com/in/fabricio-bizotto-0aa27131/)
- [Gilson Giraldi (LNCC)](https://www.lncc.br/~gilson/)
- [Mauren Luise Sguario de Andrade (UTFPR)](https://www.linkedin.com/in/mauren-louise-sguario-coelho-de-andrade-3291b731/?originalSubdomain=br)

## Agradecimentos

- [UTFPR](https://portal.utfpr.edu.br/)
- [LNCC](https://www.lncc.br/)
- [ICMBio](https://www.gov.br/icmbio/pt-br)
- [Google Earth](https://www.google.com/earth/)
- [IFC](https://www.ifc.edu.br/)

## Direitos autorais

Este projeto está licenciado sob a licença GNU GPL v3.0 - consulte o arquivo [LICENSE.md](LICENSE.md) para obter detalhes.
