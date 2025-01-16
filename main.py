from planner.planner import Planner
from evaluator.evaluator import Evaluator
from mcts.mcts import MCTS, Node
from vlm.vlm_model import VLMModel


def test_mcts_tree():
    # 初始化 Planner、VLM 和 Evaluator
    planner = Planner(model_name="google/paligemma2-3b-pt-896")
    vlm_model = VLMModel(model_name="google/paligemma2-10b-pt-896")
    evaluator = Evaluator(model_name="google/paligemma2-3b-pt-896")

    # 初始化 MCTS
    mcts = MCTS(planner=planner, vlm_model=vlm_model, result_evaluator=evaluator, num_simulations=5)

    # 创建根节点
    root_state = "Review the Histopathology image and pick the opyion the base aligns with the severity illustrated. Options: A.malignant B.benign"
    root_image_path = "/home/MCTS-PPO/image/Lok1_SG_LMP_image.png"
    root_node = Node(state=root_state, parent=None, image_path=root_image_path)

    # 执行 MCTS 搜索
    best_node = mcts.search(root_node)

    # 打印整棵树
    mcts.print_tree(root_node)

    # 打印从最佳节点到根节点的路径
    mcts.print_path(best_node)

    # 输出最佳路径
    print(f"\nBest Plan: {best_node.state}")
    print(f"Best Answer: {best_node.answer}")
    print(f"Best Reward: {best_node.get_average_reward()}")


if __name__ == "__main__":
    test_mcts_tree()
