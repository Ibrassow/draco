import numpy as np
import matplotlib.pyplot as plt 
import itertools

"""
Naive RRT
"""

class Node:

    id_iter = itertools.count()

    def __init__(self, configuration, parent_id=None):
        self.children = []
        self.configuration = configuration
        self.id = next(Node.id_iter)
        self.parent_id = parent_id

    def add_child(self, child):
        child.set_parent_id(self.id)
        self.children.append(child)


    def set_parent_id(self, parent_id):
        self.parent_id = parent_id



    def clear(self):
        self.children.clear()
        self.configuration = None






class RRTbase:
    
    def __init__(self, dim, stepsize, lb=None, ub=None):
        self.tree = []
        self.stepsize = stepsize
        self.n = dim
        self.obstacles = []

        if (lb.all()!=None) and (ub.all()!=None):
            self.lower_bounds = lb
            self.upper_bounds = ub
        else:
            print("No bounds definition for configuration - default values [0-100]")
            self.lower_bounds = [0] * self.n
            self.upper_bounds = [100] * self.n





    def start_tree(self, q):
        self.tree = []
        self.add_vertex_to_tree(q)


    def add_vertex_to_tree(self, q_node):
        # Add a Node instance
        self.tree.append(q_node)

    def add_edge_to_vertex(self, parent_node, child_node):
        # Add a Node instance to the children of the parent node
        parent_node.add_child(child_node)


    def get_plan(self, q_init, q_goal, max_iter=500):

        self.start_tree(Node(q_init))

        for k in range(max_iter):

            q_rand = self.sample_random_configuration()

            if self.check_collision(q_rand):
                continue

            idx_nearest = self.find_nearest_neighbor(q_rand)

            new_node = self.generate_new_close_configuration(q_rand, idx_nearest)

            self.add_vertex_to_tree(new_node)
            self.add_edge_to_vertex(self.tree[idx_nearest], new_node)

            if self.compute_distance(new_node.configuration, q_goal) < self.stepsize:
                # Connect to goal
                goal_node = Node(q_goal)
                self.add_vertex_to_tree(goal_node)
                self.add_edge_to_vertex(new_node, goal_node)
                print("Found path!")
                return self.tree
            
        print("Failed to find a path.")
        return self.tree
    

    def sample_random_configuration(self):

        random_config = np.random.uniform(self.lower_bounds, self.upper_bounds, self.n)
        return random_config



    def compute_distance(self, q1, q2):
        return np.linalg.norm(q1 - q2)


    def find_nearest_neighbor(self, q_goal):
        """
        Returns the array index of the nearest node in the tree 
        """
        array_config = np.array([node.configuration for node in self.tree])
        tree_idx_nearest = (np.linalg.norm(array_config - q_goal, axis=1)).argmin()
        return tree_idx_nearest


    def generate_new_close_configuration(self, q, idx_goal):
        diff = rrt.tree[idx_goal].configuration - q
        dist = self.compute_distance(rrt.tree[idx_goal].configuration, q) 
        return Node(rrt.tree[idx_goal].configuration + (self.stepsize/dist) * diff)
    
    def retrieve_path(self):
        # Only after running RRT obviously
        path = [self.tree[-1]]
        for node in reversed(self.tree):
            if node.id == path[-1].parent_id:
                path.append(node)
        path.reverse()
        return path
    
    #dummy
    def generate_2D_random_obstacles(self, qstart, qgoal, nb_obs, max_radius):
        self.obstacles = []

        while len(self.obstacles) != nb_obs:
            qobs = self.sample_random_configuration()
            r = np.random.uniform(0.01, max_radius)
            ## Ensure obstacle not on start or goal
            if self.compute_distance(qobs, qstart) < r or self.compute_distance(qobs, qgoal) < r:
                continue
            else:
                self.obstacles.append([qobs,r])

        return self.obstacles

    def check_collision(self, q):
        ## Accompanying dummy
        for o in self.obstacles:
            if np.linalg.norm(o[0] - q) < o[1]:
                return True
        return False # Everything is fine then




    def plot_rrt(self, path=[]):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.plot()
        ax.scatter(path[0].configuration[0], path[0].configuration[1],c='g',label="Start", marker='*', s=100)
        ax.scatter(path[-1].configuration[0], path[-1].configuration[1],c='b',label="Goal", marker='*', s=100)
        for node in self.tree:
            if node.parent_id is not None:
                x1, y1 = node.configuration
                x2, y2 = self.tree[node.parent_id].configuration
                ax.plot([x1, x2], [y1, y2], 'k-', lw=1, alpha=0.5)
        for node in path:
            if node.parent_id is not None:
                x1, y1 = node.configuration
                x2, y2 = self.tree[node.parent_id].configuration
                ax.plot([x1, x2], [y1, y2], 'r-', lw=2)
        
        for obs in self.obstacles:
            x, y = obs[0]
            r = obs[1]
            ax.scatter(x, y, s=50*r, c='g', edgecolors='g')
        plt.legend()
        plt.show()



if __name__ == "__main__":


    lb = np.array([1,1,1,1,1,1])*(-np.pi/2)
    ub = np.array([1,1,1,1,1,1])*(np.pi/2)
    rrt = RRTbase(6,0.5, lb=lb, ub=ub)
    q_start = np.array([0,0,0,0,0,0])
    q_goal = np.array([1,0.2,-0.4,0,0.3,-0.12])
    #rrt.generate_2D_random_obstacles(q_start, q_goal, 10, 0.2)
    import time
    t1 = time.time()
    tree = rrt.get_plan(q_start, q_goal, max_iter=100000)
    tf = time.time()
    print("Took {} sec".format(tf-t1))
    print(len(tree))
    path = rrt.retrieve_path()
    print(path)
    o = []
    for n in path:
        o.append(n.configuration)
    o = np.array(o)
    print(o)



    #rrt.plot_rrt(path)
    #plt.plot(o[:,0], o[:,1])
    #plt.show()
    """rrt = RRTbase(6,0.2,lb=([0]*6),ub=([3.14]*6))
    q_start = np.array([0.2,0.2,0.2,0.2,0.2,0.2])
    q_goal = np.array([1.7,1.8,1.4,1.4,1.2,1.3])
    tree = rrt.get_plan(q_start, q_goal, max_iter=10000)
    print(len(tree))"""



    