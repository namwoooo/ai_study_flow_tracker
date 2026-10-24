{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jvOPfY4APXCf"
      },
      "source": [
        "#LLaMA 로딩 from hugging face\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "L03ReTOnUDNR",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "02c8e886-9945-44ce-865b-deb393f36970"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "collapsed": true,
        "id": "QNwHu-OTN_ml",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000,
          "referenced_widgets": [
            "5475765c325f4fa9adc517701a683c13",
            "be1f83c9e8d64bf78232e6fbe18fec70",
            "c757bab1e6364eeb9ecf3d1ecb203acb",
            "894d593388d34f5198479251cd6538d5",
            "61e7b524b6d5435b93c8ad786b23c3b5",
            "8e0d66ab7f104246912e88dd9b7abcb2",
            "95dc6ed13c3c43e495356c4914b9e181",
            "288664ca0e3e49cd869c76e0b31969e0",
            "54b83f968a0f4ad7a2c06fea9cc9b433",
            "53a233c43a0e45289faaee32c508aacc",
            "eb891f49a54f47bab74680f5121250e4",
            "c41375569f414b338f94a42911f6a797",
            "5ed6d845770c4aa09e3e6fd68fd13be6",
            "bfd25f0807e645c49248a08a000277fe",
            "5f18e60016ee4d8bbc8613a6540ea5f7",
            "58130482373c43698d1d855c7d4496fa",
            "3358555a15644c4e9e0c9aa82c2c65dc",
            "ec1e7aea69c1487ea5a5619ab2af39b8",
            "b2ac2dc975e54da0951811228c589e12",
            "4317553ce98b4706ae40131260f22d2b",
            "6ed9bad31fa6469c9d8989fdb8d1c662",
            "2d0b677f89c548b6be5131e1b8e868ed",
            "bafebffd772b48c9a872f047b3de42f4",
            "c2b395f8a58f4eb18209cf6de3f73b91",
            "9c4490e91c93402ca33be1473ea41395",
            "ce7b3656b8c64e408686a36356cdd29b",
            "33d44075aa1641e1959cfcfc5be418cc",
            "2e5dd35e1b3f4133b80576941c86f1af",
            "8966c7904f2e4f0f8654ce03655aabfa",
            "e3afd0d29a424671891540b238bd5e1c",
            "acb5d851470a427981e627d5130073af",
            "1b7405c22b6749359bd172ff35cf0ccb",
            "356e8f1b2719468d958c0bb45865a784",
            "c96e66f3ba2c4d20a43fa750e647b08c",
            "f25fe82d0dac474cb7e13e3755fe3efa",
            "f633facd98b141918a1f8ce2f9cdc226",
            "b55b2e62dfa745668ca23f04b4b307bd",
            "32dfd85a051b42a4b014dde308cd4e6b",
            "8a4869a338154f6aa69ea49fe8e8d086",
            "9de8ba904b484e3f8a957b9a4f906c79",
            "1f72d48f9dfc424482c64dccf2835e67",
            "5a807ed0bf96484f9356c1ee8fc53f2b",
            "ca85e57bd8c747198ae7e6b625e68a33",
            "83a323f463af456f906a5cabbbd596ed",
            "cac359f15a0c49ed83588b2a300fc80f",
            "aca1c853e6774307a3ca76bc871cc051",
            "208a03a802fa404889984864981dc6dd",
            "a5bc6db85fd048dd92974a273719471f",
            "9e80aa391ac9423e813ae525b2dfc622",
            "9a03e20778e14fc0ba9f82f901b83736",
            "3fd6810b895c43359a27baf3c8794a23",
            "e6b8dd9ae2124b1a98755dcbba33491c",
            "c5f2e08c6baf44e9b5d7f54a8df2cf0f",
            "2d44bc4677824b6789eec2d7d1c7cfa0",
            "35782e5387b0449ab64d63eac3e049b6",
            "5858905718914f1eb74fd2ef20501638",
            "e6ce18c3e56048b190f50055cf3c06df",
            "86c2be9c76634f6fa7654c7238a143ef",
            "10238ab8028d4f4ab3aa916950e59691",
            "aa4142713b8e4df9b01bd9d92d51d98e",
            "8c19614aa8984544a97e851fe750a075",
            "05eaeb45dd3d4c419b6f27b39585fc86",
            "7e335b2537564947af2061adbdf367fc",
            "f2edeeb21edb4f3e9593ba088d308a26",
            "1ff04396526144b387f44c7ad28b5a66",
            "a2550abc92f54a7b970c7a2b53233ce4",
            "830e870f84514cb98d79fbeee24c850b",
            "b6f1d3288d2b439ab2d446ed64f6f909",
            "bcbd90a6da54468295e29529ce639079",
            "c1f93d1400d14dd4991ba55ae7b61e27",
            "598c1a37e4cc4ae0abd2efd632f5045f",
            "ce1c4b1f699b4b5aae10bed39dce18e1",
            "e3419efdab0a443f8c3091243f849558",
            "2e5499243f8d41a8a6171b9853197922",
            "a9525f9d8850407e87d90420a3fd5ac6",
            "0efa6a7a0e084fc5a0b5ad7628d917ac",
            "516d319a823e46288994799195057509",
            "ba5b326b870940efbb77d630dce0f4df",
            "72c769f0891341de978357fff34b2242",
            "e787ae5e164c4fbabb2b1ac94638b87f",
            "932367e18dd64dc4af0a9f33a126abf0",
            "ee399845a97d4bcd9775d755f22d5a2c",
            "148ff689c954468a842177e8b4583763",
            "3ff3f798f3a84799919e3954a46c0cae",
            "6d12902606c1405f9d2253c67e853514",
            "101db05e4c754cb38e3c6e42df04964e",
            "01d3b1643ce34f8780480cf5ea1b61a2",
            "66b2962148da4baaa91b4dd7def4fdff",
            "0c43fa5479fb4d9e94af84a374f1197a",
            "1d62ec0fdd264fb6bfc981c98b75578f",
            "275ce9136c1c4d15a1f4e9ad4fe60bd9",
            "bb7a4879d40145c9bd050467fce18482",
            "35b5f2fea0924dcabfd8f74ed1ef40e5",
            "532cd9e364bc4b4eab6a2d383082b96f",
            "beb87004bd2b410e89ae226dd179aac9",
            "9ccbe0c3b4d743e2add6ed4c9aeb9b12",
            "a3147ac466214bcf96ad1e8ae5b11fed",
            "4593b7f1aaec42e5b136896028a53a1e",
            "b9a43f742d7e41e5865fc5a349eb7af6",
            "952d9c22b3c84257b39a5544b87e2fb1",
            "f10f1145c9724b158686a59d91757320",
            "4d8a191078954ffcb1a544befb8378c3",
            "eb10e00b385040bf99e8214b81fe4535",
            "72d87755d1874ee8b757db280be27b24",
            "35a8550590f843c0936f11278bbda068",
            "33c5f7d5249245c0b86f98c706d2e12a",
            "e18ec6a423fb46fb842c26da48d852f4",
            "f361bb1fe163433591d754407ca47b55",
            "a083f90914a84fab834489619b2dd9c6",
            "b84c89e120b34937b3acaebbd9264782",
            "7c09063053b949079def1d5d1d385bf0",
            "837fee452ba34957821a9c2f240c7d8b",
            "18252f82fd314d7eabf0495f41a8cd73",
            "21261f310c794aeb85692d4c5b47299f",
            "cf698130243f4b0995d25eda8f609d7c",
            "e6da1ab27fc3401a90463c46d3f446e1",
            "aa13676ad1b942199bf43c40ee4dcb32",
            "5266ebe946304a578914e55db0cc3e7e",
            "4c09b8de4ef5486396eed69b69efd5e2",
            "9f72e3aa031e4710b4c5536922e0b13c",
            "7d8266b1dce048d9a23023ef7cc5895f",
            "fecc7c0970c5455380317982cba2e1e5",
            "4f2b57424096431b9e38efed6d46a9a6",
            "12d01959e46b4fc2b2b73ddf0a1fa5fc",
            "7be5c3c94d6043fab17b5bff80caad35",
            "46986eb6985e4145b682f5c06e8ba049",
            "62269efe2ed24c5ab5aad90e6bbe09e4",
            "ec09e0588ede4e71bff1ebc72950a5c9",
            "d3f8d7bcb5a649e78fd8c2fca1b1f3ba",
            "6308957f470a46468ae3dab5736b1d90",
            "28c9ad9051c74643af02e4aa240d037a",
            "5339fcfd5c4a4eadbade1cc87aa4c641",
            "8f57b3b7eab744539ff99f69e61bfad7",
            "cd50c01285024f7f91f40ddec6c7bfd7",
            "ffa37a18ec144d529112aea8bcf702ea",
            "cb3de7815c6b45d2979752ba0e74e353",
            "d3040ba157be47fbaf7e030590cd0d7f",
            "331c973c6062424fa3ea2be57f6510d3",
            "f773ec9e166b49b2a904e8d53396bf71",
            "94fffc4ae0964dcc8fe49d6c00ba033b",
            "a2c9f02b5a3c46cb9d5054c0da5f990f",
            "7f5eff82cc1f447aa81032fcccab10ec",
            "a2453d2600254b35a97f63800e9776a1",
            "80bac6670ccc48eeb48f5fcf4272877d",
            "df74f071d3e24e7cb1a245016699a22d",
            "bc841055a5684c85aef324a3dbd328f2",
            "e0af04a1bb504538ba364777bbe2a598",
            "2ac1515cb0ea4ba98d406f9bf4501821",
            "5a0d6b5eeddf41078807f97a704c155e"
          ]
        },
        "outputId": "97ac363a-da9a-4458-f010-417ce4835a53"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: transformers in /usr/local/lib/python3.12/dist-packages (4.57.2)\n",
            "Requirement already satisfied: accelerate in /usr/local/lib/python3.12/dist-packages (1.12.0)\n",
            "Requirement already satisfied: sentencepiece in /usr/local/lib/python3.12/dist-packages (0.2.1)\n",
            "Requirement already satisfied: bitsandbytes in /usr/local/lib/python3.12/dist-packages (0.48.2)\n",
            "Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from transformers) (3.20.0)\n",
            "Requirement already satisfied: huggingface-hub<1.0,>=0.34.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.36.0)\n",
            "Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2.0.2)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (25.0)\n",
            "Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from transformers) (6.0.3)\n",
            "Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.12/dist-packages (from transformers) (2025.11.3)\n",
            "Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from transformers) (2.32.4)\n",
            "Requirement already satisfied: tokenizers<=0.23.0,>=0.22.0 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.22.1)\n",
            "Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.12/dist-packages (from transformers) (0.7.0)\n",
            "Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.12/dist-packages (from transformers) (4.67.1)\n",
            "Requirement already satisfied: psutil in /usr/local/lib/python3.12/dist-packages (from accelerate) (5.9.5)\n",
            "Requirement already satisfied: torch>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from accelerate) (2.9.0+cpu)\n",
            "Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (2025.10.0)\n",
            "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (4.15.0)\n",
            "Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface-hub<1.0,>=0.34.0->transformers) (1.2.0)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate) (75.2.0)\n",
            "Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate) (1.14.0)\n",
            "Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate) (3.6)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch>=2.0.0->accelerate) (3.1.6)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->transformers) (2025.11.12)\n",
            "Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch>=2.0.0->accelerate) (1.3.0)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch>=2.0.0->accelerate) (3.0.3)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "VBox(children=(HTML(value='<center> <img\\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "5475765c325f4fa9adc517701a683c13"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/tokenization_auto.py:1041: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer_config.json:   0%|          | 0.00/55.4k [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "ec1e7aea69c1487ea5a5619ab2af39b8"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "tokenizer.json:   0%|          | 0.00/9.09M [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "8966c7904f2e4f0f8654ce03655aabfa"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "special_tokens_map.json:   0%|          | 0.00/296 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "9de8ba904b484e3f8a957b9a4f906c79"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/transformers/models/auto/auto_factory.py:492: FutureWarning: The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "config.json:   0%|          | 0.00/855 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "3fd6810b895c43359a27baf3c8794a23"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "`torch_dtype` is deprecated! Use `dtype` instead!\n",
            "/usr/local/lib/python3.12/dist-packages/jax/_src/cloud_tpu_init.py:86: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c \"echo always > /sys/kernel/mm/transparent_hugepage/enabled\")\n",
            "  warnings.warn(\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model.safetensors.index.json:   0%|          | 0.00/23.9k [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "05eaeb45dd3d4c419b6f27b39585fc86"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Fetching 4 files:   0%|          | 0/4 [00:00<?, ?it/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "e3419efdab0a443f8c3091243f849558"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model-00003-of-00004.safetensors:   0%|          | 0.00/4.92G [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "3ff3f798f3a84799919e3954a46c0cae"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model-00001-of-00004.safetensors:   0%|          | 0.00/4.98G [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "beb87004bd2b410e89ae226dd179aac9"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model-00002-of-00004.safetensors:   0%|          | 0.00/5.00G [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "33c5f7d5249245c0b86f98c706d2e12a"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "model-00004-of-00004.safetensors:   0%|          | 0.00/1.17G [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "aa13676ad1b942199bf43c40ee4dcb32"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "ec09e0588ede4e71bff1ebc72950a5c9"
            }
          },
          "metadata": {}
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "generation_config.json:   0%|          | 0.00/184 [00:00<?, ?B/s]"
            ],
            "application/vnd.jupyter.widget-view+json": {
              "version_major": 2,
              "version_minor": 0,
              "model_id": "f773ec9e166b49b2a904e8d53396bf71"
            }
          },
          "metadata": {}
        }
      ],
      "source": [
        "!pip install transformers accelerate sentencepiece bitsandbytes\n",
        "from huggingface_hub import notebook_login\n",
        "notebook_login()\n",
        "\n",
        "import torch\n",
        "from transformers import AutoTokenizer, AutoModelForCausalLM\n",
        "\n",
        "LLM_NAME = \"meta-llama/Llama-3.1-8B-Instruct\"\n",
        "\n",
        "tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, use_auth_token=True)\n",
        "llm = AutoModelForCausalLM.from_pretrained(\n",
        "    LLM_NAME,\n",
        "    torch_dtype=torch.float16,\n",
        "    device_map=\"auto\",\n",
        "    use_auth_token=True\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZVyVkA3O1-yU"
      },
      "source": [
        "#7개 feature segment 분석 / 프롬프트"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Z33YLMK-T1jd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ee3fcb51-d5ca-4418-9ecb-19db36ad64ec"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[세그먼트 0] (약 0~5초)\n",
            "- final: 68.9점\n",
            "- phone_face: 0.50, head_angle: 0.07, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 1.00, perclos: 1.00\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 1] (약 5~10초)\n",
            "- final: 52.8점\n",
            "- phone_face: 0.50, head_angle: 0.55, shoulder_tilt: 0.00, hand_face: 0.99\n",
            "- blink: 0.64, perclos: 0.00\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 2] (약 10~15초)\n",
            "- final: 47.4점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.71, perclos: 0.00\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 3] (약 15~20초)\n",
            "- final: 52.2점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.64, perclos: 0.37\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 4] (약 20~25초)\n",
            "- final: 57.0점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 1.00, perclos: 0.31\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 5] (약 25~30초)\n",
            "- final: 64.1점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.93, perclos: 0.83\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 6] (약 30~35초)\n",
            "- final: 64.1점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.93, perclos: 0.83\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 7] (약 35~40초)\n",
            "- final: 65.3점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 1.00, perclos: 0.83\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 8] (약 40~45초)\n",
            "- final: 62.3점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.93, perclos: 0.71\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 9] (약 45~50초)\n",
            "- final: 47.4점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.57, perclos: 0.14\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 10] (약 50~55초)\n",
            "- final: 64.1점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.93, perclos: 0.83\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 11] (약 55~60초)\n",
            "- final: 65.0점\n",
            "- phone_face: 0.50, head_angle: 0.00, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.93, perclos: 0.89\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 12] (약 60~65초)\n",
            "- final: 49.5점\n",
            "- phone_face: 0.50, head_angle: 0.27, shoulder_tilt: 0.00, hand_face: 1.00\n",
            "- blink: 0.64, perclos: 0.00\n",
            "- 상태: 정상\n",
            "\n",
            "[세그먼트 13] (약 65~70초)\n",
            "- final: 53.8점\n",
            "- phone_face: 0.50, head_an\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "def build_focus_summary(csv_path):\n",
        "    df = pd.read_csv(csv_path)\n",
        "\n",
        "    lines = []\n",
        "    for _, row in df.iterrows():\n",
        "        seg = int(row[\"segment_idx\"])\n",
        "        final = row[\"final\"]\n",
        "\n",
        "        phone_face = row[\"phone_face\"]\n",
        "        head_angle = row[\"head_angle\"]\n",
        "        shoulder = row[\"shoulder_tilt\"]\n",
        "        hand_face = row[\"hand_face\"]\n",
        "        blink = row[\"blink\"]\n",
        "        perclos = row[\"perclos\"]\n",
        "        seat = bool(row[\"seat_departed\"])\n",
        "\n",
        "        status = \"자리 이탈 발생\" if seat else \"정상\"\n",
        "\n",
        "        line = (\n",
        "            f\"[세그먼트 {seg}] (약 {seg*5}~{(seg+1)*5}초)\\n\"\n",
        "            f\"- final: {final*100:.1f}점\\n\"\n",
        "            f\"- phone_face: {phone_face:.2f}, head_angle: {head_angle:.2f}, \"\n",
        "            f\"shoulder_tilt: {shoulder:.2f}, hand_face: {hand_face:.2f}\\n\"\n",
        "            f\"- blink: {blink:.2f}, perclos: {perclos:.2f}\\n\"\n",
        "            f\"- 상태: {status}\\n\"\n",
        "        )\n",
        "        lines.append(line)\n",
        "\n",
        "    return \"\\n\".join(lines)\n",
        "\n",
        "summary_text = build_focus_summary(\"/content/drive/MyDrive/ai_study_flow_tracker/data_preprocessing/labels_revised.csv\")\n",
        "print(summary_text[:2000])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QMGLO4fvESbA"
      },
      "outputs": [],
      "source": [
        "# 시각화\n",
        "\n",
        "import re\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def parse_focus_segments(summary_text):\n",
        "    \"\"\"\n",
        "    [세그먼트 N] (약 X~Y초) - final: xx.x점 형태의 텍스트를\n",
        "    [(start_time, final_score)] 리스트로 변환\n",
        "    \"\"\"\n",
        "    pattern = r\"\\[세그먼트 (\\d+)\\] \\(약 (\\d+)~(\\d+)초\\)\\s*- final: ([\\d\\.]+)점\"\n",
        "    matches = re.findall(pattern, summary_text)\n",
        "    times_scores = []\n",
        "    for seg_idx, start, end, score in matches:\n",
        "        # x축을 초 단위로 - 구간의 시작(평균도 무방)\n",
        "        times_scores.append((int(start), float(score)))\n",
        "    return times_scores\n",
        "\n",
        "def save_and_show_final_score_chart(scores, save_path=\"/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/results/final_score_chart.png\"):\n",
        "    # 그래프에서는 분으로 표현\n",
        "    times = [t//60 for t, _ in scores]\n",
        "    values = [v for _, v in scores]\n",
        "\n",
        "    plt.figure(figsize=(12, 5))\n",
        "    plt.plot(times, values, marker='', color='dodgerblue', label='final score')\n",
        "    plt.xlabel('Time (min)')\n",
        "    plt.ylabel('Score')\n",
        "    plt.title('Focus Score through Time')\n",
        "    plt.grid(True)\n",
        "    plt.legend()\n",
        "    if values:\n",
        "        max_idx = values.index(max(values))\n",
        "        min_idx = values.index(0)\n",
        "        plt.legend()\n",
        "    plt.tight_layout()\n",
        "    plt.savefig(save_path)\n",
        "    plt.close()\n",
        "    print(f\"그래프 저장 완료: {save_path}\")\n",
        "\n",
        "    # Jupyter/Colab에서 바로 띄워주기\n",
        "    try:\n",
        "        from IPython.display import Image, display\n",
        "        display(Image(filename=save_path))\n",
        "    except ImportError:\n",
        "        pass"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fdcPpz8lEWtG"
      },
      "outputs": [],
      "source": [
        "times_scores = parse_focus_segments(summary_text)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rn3xoWe4Esuq"
      },
      "outputs": [],
      "source": [
        "save_and_show_final_score_chart(times_scores, save_path=\"/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/results/final_score_chart.png\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# 프롬프트\n",
        "\n",
        "def build_llama_prompt(segments_json: str, user_name: str = \"사용자\") -> str:\n",
        "    \"\"\"\n",
        "    segments_json: 세그먼트별 점수 리스트(JSON)\n",
        "    baseline_json: 개인 baseline(JSON)\n",
        "    \"\"\"\n",
        "    return f\"\"\"\n",
        "[SYSTEM]\n",
        "당신은 학생들의 학습 집중 패턴을 분석하고,\n",
        "누구나 쉽게 이해할 수 있는 \"친절한 한국어 설명 리포트\"를 작성하는 전문가입니다.\n",
        "\n",
        "목표:\n",
        "- 어려운 기술 용어나 숫자를 가능한 한 사용하지 않고,\n",
        "- 학생이 읽으면 “아, 내가 공부할 때 이런 습관이 있구나” 하고 직관적으로 이해할 수 있도록 설명하세요.\n",
        "- 행동을 중심으로 설명하고, 개선 방향은 실제로 따라 할 수 있는 것만 제시하세요.\n",
        "- 어려운 용어(EAR, landmark, baseline 등)는 절대 사용하지 않습니다.\n",
        "- 점수나 수치를 그대로 말하지 말고 “자주 ~했다”, “대체로 안정적이었다”, “중간에 흐트러짐이 있었다”처럼 자연스러운 언어로 설명하세요.\n",
        "\n",
        "[INPUT DATA]\n",
        "\n",
        "1) 세그먼트 데이터 (5초 단위)\n",
        "- segment_idx: 번호\n",
        "- phone_face, seat_depart, head_angle, shoulder_tilt,\n",
        "  hand_face, blink, perclos, final, personalized 등은\n",
        "  \"학습 중 어떤 행동이 얼마나 안정적이었는지\" 나타내는 값입니다.\n",
        "- 하지만 리포트에서는 절대 값이나 숫자를 직접 말하지 말고,\n",
        "  행동/상황을 자연스럽게 해석해서 전달하세요.\n",
        "\n",
        "세그먼트 데이터:\n",
        "{segments_json}\n",
        "\n",
        "[ASSISTANT TASK]\n",
        "위 데이터를 기반으로 {user_name}의 학습 집중 패턴을\n",
        "학생이 읽어도 이해될 만큼 쉬운 리포트로 작성해주세요.\n",
        "\n",
        "반드시 다음 구조를 따르세요:\n",
        "\n",
        "---\n",
        "\n",
        "[1. 전체 요약]\n",
        "- 전체적인 집중 흐름을 간단하게 설명합니다.\n",
        "- 초반/중반/후반 학습 흐름이 어땠는지 “느낌” 위주로 설명합니다.\n",
        "- 이 학생이 가진 좋은 점 1~2가지,\n",
        "  그리고 쉽게 흐트러지는 부분 1~2가지를 부드럽게 언급합니다.\n",
        "\n",
        "---\n",
        "\n",
        "[2. 시간 흐름에 따른 집중 패턴]\n",
        "- 초반, 중반, 후반으로 나누어 어떤 특징이 있었는지 설명합니다.\n",
        "- 집중이 잘 되었던 구간은 “집중이 잘 잡힌 편”, “습관이 안정적이었다” 등으로 표현합니다.\n",
        "- 집중이 떨어진 구간은 “잠깐 흐트러짐이 있었다”, “몸이 조금 불편해 보였다” 등 자연어로 표현합니다.\n",
        "- 자리를 벗어난 구간이 있다면, “중간에 잠깐 자리에서 움직인 것으로 보입니다” 정도로 설명합니다.\n",
        "\n",
        "---\n",
        "\n",
        "[3. 주요 습관 분석 (학생이 이해하기 쉽게)]\n",
        "아래 항목을 각각 4~6문장으로 설명하세요.\n",
        "절대 기술적 용어/수치 X.\n",
        "\n",
        "● 시선 습관\n",
        "- 화면이나 교재를 얼마나 안정적으로 바라보는지\n",
        "- 흔들렸던 순간이 있었다면 어떤 흐름 때문인지\n",
        "\n",
        "● 고개·목 자세\n",
        "- 고개가 자주 움직였는지\n",
        "- 집중이 잘 될 때의 머리 움직임 특징\n",
        "\n",
        "● 어깨·상체 자세\n",
        "- 상체가 안정적이었는지\n",
        "- 긴장/피로로 인한 변화가 있었는지\n",
        "\n",
        "● 손·얼굴의 움직임\n",
        "- 손이 얼굴로 자주 갔는지\n",
        "- 산만해지는 타이밍과의 관계\n",
        "\n",
        "● 눈 깜빡임·피곤함 신호\n",
        "- 중간에 피곤했을 가능성이 있었는지\n",
        "- 집중 흐름과 피로 신호의 상관성\n",
        "\n",
        "---\n",
        "\n",
        "[4. {user_name}의 집중 스타일 해석]\n",
        "- 이 학생이 어떤 방식으로 집중하는지 1~2단락으로 설명하세요.\n",
        "- 예: \"초반에 빠르게 몰입하지만 일정 시간이 지나면 몸이 조금씩 움직이며 집중력이 떨어지는 편입니다.\"\n",
        "- 또는 \"시선은 안정적이지만, 손이 얼굴 쪽으로 자주 가면서 흐름이 끊기는 경향이 있습니다.\" 등.\n",
        "\n",
        "---\n",
        "\n",
        "[5. 맞춤형 개선 팁 (실제로 할 수 있는 것만)]\n",
        "아래 예시처럼 구체적이고 간단한 행동 조언 4~7개를 bullet로 제시하세요.\n",
        "- “고개가 자주 움직이면 10분에 한 번씩 상체를 한 번 펴 주세요.”\n",
        "- “손이 얼굴로 자주 간다면 펜을 손에 쥐고 공부하면 도움이 됩니다.”\n",
        "- “잠깐 피곤해 보였다면 20~30분 간격으로 10초 정도 눈 스트레칭을 해보세요.”\n",
        "\n",
        "절대 모호한 조언 금지(예: “더 집중하세요”, “자세를 바로 하세요”).\n",
        "\n",
        "---\n",
        "\n",
        "[6. 마무리]\n",
        "- 이 학생이 가진 강점을 다시 한 번 강조하고,\n",
        "- 앞으로 어떻게 더 좋아질 수 있는지 긍정적인 톤으로 마무리하세요.\n",
        "\n",
        "\"\"\"\n"
      ],
      "metadata": {
        "id": "VuNCYzqultL1"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "9Lm_K2D52eyL"
      },
      "source": [
        "#리포트 생성"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uVwMh3KeUXut",
        "outputId": "b698e5a6-59fb-42d0-9f0a-555f1c74eea9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "---\n",
            "\n",
            "[7. 참고]\n",
            "- 학생이 리포트를 읽고 어떤 행동을 개선해야 하는지\n",
            "- 그 행동을 실천할 수 있는 이유와 방법을 간단하게 설명해 주세요.\n",
            "\n",
            "---\n",
            "\n",
            "---\n",
            "\n",
            "[1. 전체 요약]\n",
            "김이화 학생의 학습 집중 패턴은 전체적으로 안정적이었지만, 중반에 약간의 흐트러짐이 있었다. 초반에는 집중이 잘 잡혔고, 후반에는 시선이 안정적이었다. 좋은 점으로는 시선 습관이 안정적이고, 고개 자세가 좋았음을 알 수 있다. 하지만, 중반에 잠깐 흐트러짐이 있었고, 손이 얼굴 쪽으로 자주 갔다는 점이 약간의 흐름 끊김을 보였다. \n",
            "\n",
            "---\n",
            "\n",
            "[2. 시간 흐름에 따른 집중 패턴]\n",
            "초반 0~30초 동안 집중이 잘 잡혔고, 시선 습관이 안정적이었다. 중반 30~60초 동안 중간에 약간의 흐트러짐이 있었고, 손이 얼굴 쪽으로 자주 갔다. 후반 60~100초 동안 시선이 안정적이었지만, 중간에 잠깐 피곤해 보였다. \n",
            "\n",
            "---\n",
            "\n",
            "[3. 주요 습관 분석 (학생이 이해하기 쉽게)]\n",
            "시선 습관: 화면이나 교재를 안정적으로 바라보았고, 흔들리지 않았다. 중간에 약간의 흐트러짐이 있었지만, 시선 습관이 안정적이었다.\n",
            "\n",
            "고개 자세: 고개가 자주 움직이지 않았고, 집중이 잘 될 때의 머리 움직임 특징은 안정적이었다.\n",
            "\n",
            "어깨 자세: 상체가 안정적이었고, 긴장/피로로 인한 변화가 없었다.\n",
            "\n",
            "손·얼굴의 움직임: 손이 얼굴 쪽으로 자주 갔지만, 집중 흐름과 피로 신호의 상관성이 명확하지 않다.\n",
            "\n",
            "눈 깜빡임·피곤함 신호: 중간에 피곤했을 가능성이 있었지만, 집중 흐름과 피로 신호의 상관성이 명확하지 않다.\n",
            "\n",
            "---\n",
            "\n",
            "[4. 김이화의 집중 스타일 해석]\n",
            "김이화 학생은 초반에 빠르게 몰입하지만 일정 시간이 지나면 몸이 조금씩 움직이며 집중력이 떨어지는 편입니다. 시선은 안정적이지만, 손이 얼굴 쪽으로 자주 가면서 흐름이 끊기는 경향이 있습니다.\n",
            "\n",
            "---\n",
            "\n",
            "[5. 맞춤형 개선 팁]\n",
            "- 고개가 자주 움직이면 10분에 한 번씩 상체를 한 번 펴 주세요.\n",
            "- 손이 얼굴로 자주 간다면 펜을 손에 쥐고 공부하면 도움이 됩니다.\n",
            "- 잠깐 피곤해 보였다면 20~30분 간격으로 10초 정도 눈 스트레칭을 해보세요.\n",
            "- 시선이 안정적이지만, 손이 얼굴 쪽으로 자주 가는 경우, 책상 위에 작은 물건을 두고 손이 그 물건을 찾을 때까지 움직일 수 있도록 하세요.\n",
            "- 중간에 약간의 흐트러짐이 있었으니, 중간중간에 5분간의 짧은 휴식 시간을 갖는 것이 도움이 될 수 있습니다.\n",
            "\n",
            "---\n",
            "\n",
            "[6. 마무리]\n",
            "김이화 학생은 시선 습관이 안정적이고, 고개 자세가 좋았습니다. 앞으로도 이러한 습관을 유지하고, 중간중간에 휴식 시간을 가질 수 있도록 노력하면 더 좋은 학습 집중력을 기대할 수 있습니다.\n",
            "\n",
            "---\n",
            "\n",
            "[7. 참고]\n",
            "학생이 리포트를 읽고, 시선 습관이 안정적이고 고개 자세가 좋다는 점을 반영하여 공부를 할 수 있습니다. 또한, 중간에 약간의 흐트러짐이 있었으니, 중간중간에 휴식 시간을 가질 수 있도록 노력하면 더 좋은 학습 집중력을 기대할 수 있습니다. 이와 같은 행동을 실천하면 학습 성과가 향상될 것입니다.  김이화 학생은 시선 습관이 안정적이고, 고개 자세가 좋았습니다. 앞으로도 이러한 습관을 유지하고, 중간중간에 휴식 시간을 가질 수 있도록 노력하면 더 좋은 학습 집중력을 기대할 수\n"
          ]
        }
      ],
      "source": [
        "prompt = build_llama_prompt(summary_text, \"김이화\")\n",
        "\n",
        "inputs = tokenizer(prompt, return_tensors=\"pt\").to(llm.device)\n",
        "\n",
        "output = llm.generate(\n",
        "    **inputs,\n",
        "    max_new_tokens=1000,\n",
        "    temperature=0.6,\n",
        "    top_p=0.9\n",
        ")\n",
        "\n",
        "report = tokenizer.decode(output[0], skip_special_tokens=True)\n",
        "print(report.replace(prompt, \"\"))\n",
        "\n",
        "prompt_replaced_report = report.replace(prompt, \"\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GXdlAmzhUpVg"
      },
      "outputs": [],
      "source": [
        "with open(\"/content/drive/MyDrive/ai_study_flow_tracker/LLaMA/reports/report.txt\", \"w\", encoding=\"utf-8-sig\") as f:\n",
        "    f.write(prompt_replaced_report)"
      ]
    }
  ]
}