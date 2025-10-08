from tsfb import PipelineExecutor

pipeline = PipelineExecutor(config_path="./tsfb_configs/use_case.yaml")
pipeline.run_and_select_best(metric="mae", mode="auto")
