# qwen3\.5\-4B模型版本

# 量化版本

- 模型存放地址：

    - 存放主机：AGX Orin

    - IP地址：192\.168\.0\.118

    - 账户：orin

    - 密码：jetson

    - 文件所在目录：/home/orin/VLM/qwen3\.5/weights。（包含gguf，hf，safetensors）

- 不同版本在NX上的表现：（qwen3\.5\-4b\-mixed\-3\.1为最新一版量化）

    - PPL越小说明模型输出效果越好

    |    **模型**|    **Prefill**<br>    **\(token/s\)**|    **Decode**<br>    **\(token/s\)**|    **CUDA峰值**<br>    **\(GiB\)**|    **PPL**|    **PPL增幅**|
    |---|---|---|---|---|---|
    |    qwen3\.5\-4b\-f16|    526\.66|    9\.44|    8\.629|    9\.9484|    0%|
    |    qwen3\.5\-4b\-q4km|    551\.41|    17\.40|    3\.145|    10\.3912|    \+4\.45%|
    |    qwen3\.5\-4b\-iq3xxs|    558\.66|    19\.87|    2\.315|    11\.3121|    \+13\.71%|
    |    qwen3\.5\-4b\-iq2xss|    566\.50|    20\.45|    1\.974|    16\.9669|    \+70\.55%|
    |    qwen3\.5\-4b\-reopt\-2\.5|    486\.09|    18\.95|    1\.850|    15\.2089|    \+52\.88%|
    |    qwen3\.5\-4b\-mixed\-3\.1|    555\.06|    21\.02|    2\.164|    11\.7224|    \+17\.83%|





# 训练版本

- 模型存放地址：

    - 存放主机：RTX 3090

    - IP地址：192\.168\.0\.153

    - 账户：asus

    - 密码：Abc123

    - 所有版本所在路径：/home/asus/training\-model/



- 版本号：qwen3\.5\-4b\-lora\-v1

- 文件所在目录：/home/asus/training\-model/qwen3\.5\-4b\-lora/（包含权重、tokenizer 和配置文件等）





# Photobench整理benchmark data

- 模型存放地址：

    - 存放主机：RTX 3090

    - IP地址：[192\.168\.0\.153](http://192.168.0.153/)

    - 账户：asus

    - 密码：Abc123

- 文件所在目录：/home/asus/xy/photobench\_benchmark/



