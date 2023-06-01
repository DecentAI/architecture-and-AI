import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def make_graph(x : np.ndarray , 
               y : np.ndarray,
               threshold: int = 2) -> nx.Graph:
    '''
    args 
        x : 1차원 x좌표
        y : 1차원 x좌표
    '''
    # 거리 기반 그래프 객체 생성
    points = np.column_stack((x, y))
    distances = np.sqrt(((points[:, None, :] - points) ** 2).sum(axis=2))  # 각 좌표간의 거리 계산
    distances[distances > threshold] = np.inf  # 거리가 2보다 큰 경우 무한대로 설정 (두 좌표간의 연결을 끊음)
    G = nx.Graph()  # 그래프 객체 생성
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distances[i, j] != np.inf:
                G.add_edge(i, j, weight=distances[i, j])  # 두 좌표간의 연결 추가
    return G