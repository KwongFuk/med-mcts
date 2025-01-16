import numpy as np
import math


class MCTS:
    def __init__(self, planner, vlm_model, result_evaluator, num_simulations=100, exploration_constant=1.414):
        """
        初始化 MCTS
        """
        self.planner = planner
        self.vlm_model = vlm_model
        self.result_evaluator = result_evaluator
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

    def print_tree(self, root):
        """
        打印整个 MCTS 树
        :param root: 根节点
        """
        print("\nMCTS Tree Structure:")
        root.print_tree()

    def search(self, root):
        print("search")

        for i in range(self.num_simulations):
            node = root
            print(f"Simulation {i + 1}/{self.num_simulations}: Starting from root.")

            # Selection
            node = self.selection(node)
            print(f"Selected node: {node.state}")

            # Expansion
            if node.is_empty():
                print(f"Expanding node: {node.state}")
                self.expansion(node)
            else:
                print(f"Node {node.state} is not empty. Skipping expansion.")

            # Evaluation
            if node.parent is not None:  # 根节点没有父节点，跳过评估
                print(f"Evaluating node: {node.state}")
                try:
                    reward = self.evaluation(node)
                    print(f"Evaluation reward: {reward}")
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    continue

                # Backpropagation
                print(f"Backpropagating reward: {reward} for node: {node.state}")
                self.backpropagation(node, reward)

        # 返回得分最高的子节点
        if not root.children:
            raise ValueError("No children were generated for the root node during the simulations.")
        best_node = max(root.children, key=lambda n: n.get_average_reward())
        print(f"Best node selected: {best_node.state}")
        return best_node


    def selection(self, node):
        print("selection")
        """
        Selection: 选择最有希望的叶子节点
        :param node: 当前节点
        :return: 最佳叶子节点
        """
        while node.children:
            node = max(
                node.children,
                key=lambda child: self.uct(child)
            )
        return node

    def uct(self, node):
        print("uct")
        """
        计算 UCT 值
        :param node: 当前节点
        :return: UCT 值
        """
        if node.visits == 0:
            return float('inf')  # 优先探索未访问的节点
        return (node.reward / node.visits) + self.exploration_constant * math.sqrt(
            math.log(node.parent.visits) / node.visits
        )

    def expansion(self, node):
        print("expansion")
        """
        Expansion: 扩展当前节点，生成新的子节点
        """
        # 使用 Planner 生成两个计划

        plans = self.planner.generate_plan(
            current_state=node.state,
            reflection_material=node.get_reflection_material(),
            image_path=node.image_path
        )

        if not plans or len(plans) < 2:
            print(f"Failed to generate plans for node: {node.state}")
            return

        plan_a, plan_b = plans[0], plans[1]

        # 使用 VLM 根据 Plan A 和 Plan B 生成回答
        answer_a = self.vlm_model.predict(node.image_path, plan_a)
        answer_b = self.vlm_model.predict(node.image_path, plan_b)

        if not answer_a or not answer_b:
            print(f"Failed to generate answers for node: {node.state}")
            return

        # 使用 Evaluator 对两个回答评分
        try:
            score_a = self.result_evaluator.evaluate(node.image_path, plan_a, answer_a)
            score_b = self.result_evaluator.evaluate(node.image_path, plan_b, answer_b)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return

        # 创建子节点并添加到当前节点
        child_node_a = Node(state=plan_a, parent=node, image_path=node.image_path)
        child_node_a.answer = answer_a
        child_node_a.reward = score_a
        node.add_child(child_node_a)

        child_node_b = Node(state=plan_b, parent=node, image_path=node.image_path)
        child_node_b.answer = answer_b
        child_node_b.reward = score_b
        node.add_child(child_node_b)

        print(f"Node expanded: {node.state} -> Children: {[child.state for child in node.children]}")


    def evaluation(self, node):
        print("evaluation")
        
        """
        Evaluation: 使用 Result Evaluator 对节点进行评估
        :param node: 当前节点
        :return: 评估得分
        """
        image_path = node.image_path  # 图像路径从节点继承
        question = node.state         # 当前问题（即节点的状态）
        candidate_answer = node.answer  # 当前节点生成的答案

        if not candidate_answer:
            raise ValueError(f"Node answer is empty; cannot evaluate. Node state: {node.state}")

        # 调用 Evaluator 评估
        reward = self.result_evaluator.evaluate(image_path, question, candidate_answer)
        node.reward = reward
        return reward

    def backpropagation(self, node, reward):
        print("backpropagation")
        """
        Backpropagation: 回溯更新节点及其祖先的平均值和访问计数
        :param node: 当前节点
        :param reward: 当前节点的奖励值
        """
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def print_path(self, best_node):
        """
        打印从最佳节点到根节点的路径
        """
        path = best_node.get_path_to_root()
        print("\nPath from Best Node to Root:")
        for depth, node in enumerate(path):
            print(f"Depth {depth}: Node(State={node.state[:30]}, Visits={node.visits}, Reward={node.reward:.2f})")


class Node:
    def __init__(self, state, parent=None, image_path=None):
        """
        MCTS 节点
        :param state: 当前节点的状态
        :param parent: 父节点
        :param image_path: 图像路径（用于多模态任务）
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward = 0.0
        self.total_reward = 0.0
        self.answer = None  # 保存当前节点的回答
        self.image_path = image_path  # 保存图像路径

    def add_child(self, child):
        """
        添加子节点
        :param child: 子节点
        """
        self.children.append(child)

    def is_empty(self):
        """
        判断是否为空节点
        :return: 布尔值
        """
        # 如果节点没有子节点，则为空节点
        if len(self.children) == 0:
            return True
    
        # 否则不是空节点
        return False

    def get_depth(self):
        """
        计算当前节点的深度
        :return: 当前节点的深度（根节点深度为 0）
        """
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def get_average_reward(self):
        """
        获取节点的平均奖励值
        :return: 平均奖励值
        """
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def get_reflection_material(self):
        """
        获取自反思材料，包括祖先、兄弟节点的信息
        :return: 自反思材料
        """
        if self.parent is None:
            # 根节点没有反思材料
            return {"ancestor_queries": [], "sibling_queries": []}

        reflection_material = {
            "ancestor_queries": [ancestor.state for ancestor in self.get_ancestors()],
            "sibling_queries": [sibling.state for sibling in self.get_siblings()],
        }
        return reflection_material

    def get_ancestors(self):
        """
        获取当前节点的所有祖先
        :return: 祖先节点列表
        """
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_siblings(self):
        """
        获取当前节点的所有兄弟节点
        :return: 兄弟节点列表
        """
        if not self.parent:
            return []
        return [child for child in self.parent.children if child is not self]

    def print_tree(self, depth=0):
        """
        递归打印树结构
        :param depth: 当前节点的深度（用于缩进表示层级）
        """
        indent = "  " * depth
        print(f"{indent}Node(State={self.state[:30]}, Visits={self.visits}, Reward={self.reward:.2f})")
        for child in self.children:
            child.print_tree(depth + 1)

    def get_path_to_root(self):
        """
        获取从当前节点到根节点的路径
        :return: 路径上的节点列表
        """
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return path[::-1]  # 反转路径，使根节点在最前



