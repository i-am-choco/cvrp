# CVRP

[中文文档](README_zh.md) | [English Documentation](README.md)

## Project Introduction

### 1. Project Goals

This project uses GA algorithm to solve the CVRP problem in an idealized scenario.

### 2. Project Implementation

#### Problem Analysis

1. CVRP is an NP problem, and GA is chosen to handle it in this project

2. This project is a theoretical scenario, so the actual geographic coordinates of the customer's location are not considered

3. Elements of the CVRP Problem

   - Number of vehicles
   - Vehicle capacity
   - Customer delivery requirements
   - Starting location of the vehicle (distribution center)
   - Path planning

4. Mathematical description

   - There is a customer coordinate data $C$={1, 2, ..., $n$}, where $C_i = (x_i, y_i)$, and each customer $i$ has a demand of $q_i$
   - A distribution center $Q$
   - $k$ vehicles with a capacity of $Q$
   - The transportation cost matrix $c_{ij}$ between two points, where $c_{ij} =\sqrt{(x_i - x_j)^2 + (y_i -   y_j)^2}$, represents the distance and cost between the two points
   - Each customer point must be served by one and only one vehicle
   - The total load of each vehicle must not exceed the capacity $Q$
   - The path of each vehicle must start and end at the distribution center
   - Calculate the minimum total transportation cost

> **We can organize the appeal content and express it in vector form**

- Distribution center coordinates $O$

- There are $n$ customer data, where $i$ is the customer number

  - Treat each customer coordinate as a two-dimensional vector, $v_i$=[$x_i, y_i$]

  - So many customers form a coordinate matrix $V$

    $$
    {V} =
    \begin{bmatrix}
    x_1 & y_1 \\
    x_2 & y_2 \\
    x_3 & y_3 \\
    \vdots & \vdots \\
    x_n & y_n
    \end{bmatrix}
    $$

- Distance matrix $D$

  $$
  {D} =
  \begin{bmatrix}
  d_{11} & d_{12} & \cdots & d_{1n} \\
  d_{21} & d_{22} & \cdots & d_{2n} \\
  \vdots & \vdots & \ddots & \vdots \\
  d_{n1} & d_{n2} & \cdots & d_{nn}
  \end{bmatrix}
  $$

  For all nodes, the distance matrix is ​​represented by
  $D_{ij} = \sqrt{\sum_{k=1}^{2} (V_{i,k} - V_{j,k})^2}$

  The distance between two coordinates is represented by $d_{ij} = \|v_i - v_j\|^2 = \sqrt{(x_i - x_j)^2 +    (y_i - y_j)^2}$

- Path representation vectorization $P$

  The total distance of a path $P$ = [$0$, $i_1$, $i_2$, $i_3$, ..., $i_n$, $0$] can be expressed as

  $$
  L(P) = \sum_{t=1}^{k} d_{i_t, i_{t+1}}
  $$

  Define path index vector $p$=[$O$, $i_1$, $i_2$, ..., $i_n$, $O$]

  Combining the distance matrix $D$ with the path index vector $p$, we have the expression

  $$
  L(P) = \sum_{t=1}^{k}D_{p_|t|,p_{t+1}|}
  $$

  Where

  $t$ represents the index of the path index vector $p$

  $p_t$ represents the $t$ th element of the path index vector $p$

  $p_{|t|}$ represents the absolute value of the $t$ th element of the path index vector $p$

- Capacity constraint vectorization $Q$

  Each customer's demand is $q$ = [$q_1$, $q_2$,$q_3$,..., $q_n$], and the vehicle capacity is $Q$.

  The total demand for path $P$ can be expressed as

  $$
  Q(P) = \sum_{i \in P} q_i
  $$

  The capacity constraint is:

  $$
  Q(P) \leq Q
  $$

### 3. Project Results

Cost-optimal solution iterative driving diagram
![Cost-optimal solution iterative driving diagram](./assets/Figure_1.png)

Coordinate diagram of the data set, with red representing distribution centers and blue representing customers
![Coordinate diagram of the data set, with red representing distribution centers and blue representing customers](./assets/Figure_2.png)

Optimal solution roadmap at the end of iteration
![Optimal solution roadmap at the end of iteration](./assets/Figure_3.png)

### 4. Project Summary

## What is the VRP problem?

The VRP problem is a classic optimization problem in the field of operations research. It is a problem of finding the shortest route that visits a set of customers and returns to the depot. The VRP problem is widely used in transportation, logistics, and supply chain management.

## VRP Problems

There are many different types of VRP problems, each with its own unique characteristics and requirements. Different VRP problems have different solutions, but all of them share a common goal of finding the shortest route that visits all customers and returns to the depot.

### 1. CVRP

In CVRP, each vehicle has a maximum capacity, and each customer has a demand. The vehicles must serve all customers without exceeding their capacity.

### 2. VRPTW

In addition to the vehicle capacity constraint, VRPTW also considers the time windows of customers. Each customer has a time range during which service can be accepted, and the vehicle must arrive at the customer's location within this time range.

### 3. MDVRP

MDVRP involves multiple depots (or distribution centers), and vehicles can start from different depots to serve customers. The goal is to find an optimal vehicle scheduling plan to minimize the total travel distance or cost.

### 4. GVRP

GVRP focuses on environmental factors such as vehicle fuel consumption and exhaust emissions. The goal is to minimize the environmental impact of vehicles while meeting customer needs.

## GA theoretical knowledge

## Application Scenarios of VRP Problems

- Logistics and Distribution
- Supply Chain Management
- Production Scheduling
- Transportation Planning
