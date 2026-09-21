################ LLMSR with API (via llm.config) ################

# oscillation 1
# python main.py --llm_config llm.config --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --exp_path ./exps --exp_name oscillator1_api


# oscillation 2
# python main.py --llm_config llm.config --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --exp_path ./exps --exp_name oscillator2_api


# bacterial-growth
# python main.py --llm_config llm.config --problem_name bactgrow --spec_path ./specs/specification_bactgrow_numpy.txt --exp_path ./exps --exp_name bactgrow_api


# stress-strain
# python main.py --llm_config llm.config --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --exp_path ./exps --exp_name stressstrain_api





################ (Local LLM section removed) ################





################ (Torch optimizer examples removed) ################


################ 动态规格（基于 CSV 表头 + 背景） ################

# 示例：使用任意 CSV（首行表头，前 n-1 列为特征，最后一列为目标），动态生成规格并运行
# python main.py --llm_config llm.config \
#                --data_csv /path/to/your.csv \
#                --background "这是我的问题背景描述。" \
#                --exp_path ./exps --exp_name your_exp_dynamic
