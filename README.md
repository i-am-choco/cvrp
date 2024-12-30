# CVRP

[中文文档](README_zh.md) | [English Documentation](README.md)

## Project Introduction

### 1. Project Goals

This project uses GA algorithm to solve the CVRP problem in an idealized scenario.

### 2. 项目实现

#### Problem Analysis

1. CVRP is an NP problem, and GA is chosen to handle it in this project

2. This project is a theoretical scenario, so the actual geographic coordinates of the customer's location are not considered

3. Elements that define the CVRP problem
- Number of vehicles
- Vehicle capacity
- Customer delivery requirements
- Starting location of the vehicle (distribution center)
- Path planning

#### GA theoretical knowledge

### 3. 项目效果

Cost-optimal solution iterative driving diagram
![Cost-optimal solution iterative driving diagram](./assets/Figure_1.png)

Coordinate diagram of the data set, with red representing distribution centers and blue representing customers
![Coordinate diagram of the data set, with red representing distribution centers and blue representing customers](./assets/Figure_2.png)

Optimal solution roadmap at the end of iteration
![Optimal solution roadmap at the end of iteration](./assets/Figure_3.png)

### 4. 项目总结

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

## Application Scenarios of VRP Problems

- Logistics and Distribution
- Supply Chain Management
- Production Scheduling
- Transportation Planning
