import sys

dataset_fullname = {
    'rjudge': 'rjudge',
    'agentharm': 'agentharm',
    'asb-sa': 'asb-safety',
    'asb-se': 'asb-security',
    'aj-l': 'AgentJudge-loose',
    'aj-s': 'AgentJudge-strict',
    'aj-sa': 'AgentJudge-safety',
    'aj-se': 'AgentJudge-security',
}

if __name__ == "__main__":
    # Execute the main script
    if len(sys.argv) >= 3:
        dataset = sys.argv[1]  # First argument
        choice = sys.argv[2]  # Second argument
        
        print(f"Dataset: {dataset}")
        print(f"Task: {choice}")
    else:
        raise ValueError("Please provide at least two command-line arguments")
    
    match choice:
        case 'preprocess':
            from .tasks.preprocess import preprocess_main
            preprocess_main(dataset, dataset_fullname[dataset])
        case 'cluster':
            from .tasks.cluster import cluster_main
            cluster_main(dataset)
        case 'demo':
            from .tasks.demo import demo_main
            from .tasks.demo_repair import demo_repair_main
            demo_main(dataset)
            demo_repair_main(dataset)
        case 'infer_emb':
            from .tasks.infer_emb import infer_emb_main
            infer_emb_main(dataset, dataset_fullname[dataset])
            pass
        case 'infer':
            from .tasks.infer import infer_main
            from .tasks.infer_fix1 import fix1_main
            from .tasks.infer_fix2 import fix2_main
            infer_main(dataset)
            fix1_main(dataset)
            fix2_main(dataset)
        case 'eval':
            from .tasks.eval import eval_main
            eval_main(dataset)
        case 'direct_eval':
            from .tasks.direct_metric import direct_metric_main
            direct_metric_main(dataset, dataset_fullname[dataset])
        case 'direct_metric':
            from .tasks.direct_metric import direct_metric_main
            direct_metric_main(dataset)
        case _:
            raise ValueError("Invalid choice..")
