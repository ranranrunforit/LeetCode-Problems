# Day 1 Matrix Related Problems
# 733. Flood Fill
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        if image[sr][sc] != newColor: 
            self.dfs(image, sr, sc, image[sr][sc], newColor);
        return image
    
    def dfs(self, image, sr, sc, color, newColor):
        if sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]) or image[sr][sc] != color: return
        image[sr][sc] = newColor
        self.dfs(image, sr + 1, sc, color, newColor)
        self.dfs(image, sr - 1, sc, color, newColor)
        self.dfs(image, sr, sc + 1, color, newColor)
        self.dfs(image, sr, sc - 1, color, newColor)

# 200. Number of Islands
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if len(grid) == 0: return 0
        
        ans = 0
        for x in range( len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == '1':
                    ans += 1
                    self.__dfs(grid, x, y, len(grid), len(grid[0]))
        return ans
    
    def __dfs(self, grid, x, y, n, m):
        if x < 0 or y < 0 or x >=n or y >= m or grid[x][y] == '0':
            return
        grid[x][y] = '0'
        self.__dfs(grid, x + 1, y, n, m)
        self.__dfs(grid, x - 1, y, n, m)
        self.__dfs(grid, x, y + 1, n, m)
        self.__dfs(grid, x, y - 1, n, m)

# Day 2 Matrix Related Problems
# 695. Max Area of Island
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
 
        maximum = 0
        for i in range (len(grid)):
            for j in range(len(grid[0])):
                 if grid[i][j]:
                    maximum = max(self.dfs(grid,i,j),maximum)
        return maximum
    def dfs(self,grid, i, j):
            if i < 0 or i>= len(grid) or  j < 0 or j >= len(grid[0]) or grid[i][j] == 0 : return 0
            grid[i][j] = 0
            return 1 + self.dfs(grid, i - 1, j) + self.dfs(grid,i, j + 1) + self.dfs(grid, i + 1, j) + self.dfs(grid, i, j - 1)

# 1254. Number of Closed Islands
'''
DFS
Traverse grid, for each 0, do DFS to check if it is a closed island;
Within each DFS, if the current cell is out of the boundary of grid, return 0; 
if the current cell value is positive, return 1; 
otherwise, it is 0, change it to 2 then recurse to its 4 neighors and return the multification of them.
'''
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        return sum(self.dfs(i, j, grid) for i, row in enumerate(grid) for j, cell in enumerate(row) if not cell) 
    
    def dfs(self, i, j, grid):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
            return 0
        if grid[i][j]:
            return 1
        grid[i][j] = 2
        return self.dfs(i, j + 1, grid) * self.dfs(i, j - 1, grid) * self.dfs(i + 1, j, grid) * self.dfs(i - 1, j, grid)
# DFS        
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        ans = 0
        for x in range( len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == 0:
                    ans += self.dfs(x, y, grid)
        return ans
    
    def dfs(self, i, j, grid):
        if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
            return 0
        if grid[i][j] > 0 :
            return 1
        grid[i][j] = 2 # it is 0
        return self.dfs(i, j + 1, grid) * self.dfs(i, j - 1, grid) * self.dfs(i + 1, j, grid) * self.dfs(i - 1, j, grid)
# DFS
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        return sum(self.fn(i, j, grid) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] == 0)
    
    def fn(self, y, x, grid):
        if  y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]):  # wall
            return False 
        if grid[y][x] != 0: return True   # seen

        grid[y][x] = 1  # aka add in {seen}
        return self.fn(y + 1, x, grid) + self.fn(y - 1, x, grid) + self.fn(y, x + 1,grid) + self.fn(y, x - 1,grid) == 4
# BFS
# For each land never seen before, BFS to check if the land extends to boundary.
# If yes, return 0, if not, return 1.
# Analysis:
# Time & space: O(m * n), where m = grid.Length, n = grid[0].Length.
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        seen_land = set()

        return sum(self.bfs(seen_land, grid, i, j) for i , row in enumerate(grid) for j, cell in enumerate(row) if not cell and (i, j) not in seen_land)
        
    def bfs(self, seen_land, grid, i, j):
        seen_land.add((i, j))
        q, ans = [(i, j)], 1
        for i, j in q:
            for r, c in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):     
                if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                    ans = 0
                elif not grid[r][c] and (r, c) not in seen_land:
                    seen_land.add((r, c))
                    q.append((r, c))
        return ans

# Day 3 Matrix Related Problems
# 1020. Number of Enclaves
'''
Traverse the 4 boundaries and sink all non-enclave lands by DFS, then count the remaining.
Analysis:
Time: O(m * n), space: O(m * n)
'''
class Solution(object):
    def numEnclaves(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if r in (0, m - 1) or c in (0, n - 1):
                    self.dfs(grid, r, c)
        return sum(map(sum, grid))
    def dfs(self, grid, r, c) :
        if len(grid) > r >= 0 <= c < len(grid[0]) and grid[r][c] == 1:
            grid[r][c] = 0
            for i, j in (r, c + 1), (r, c - 1), (r + 1, c), (r - 1, c):
                self.dfs(grid, i, j)

class Solution(object):
    def numEnclaves(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if r in (0, m - 1) or c in (0, n - 1):
                    self.dfs(grid, r, c)
        return sum(map(sum, grid))
    def dfs(self, grid, r, c) :
        if len(grid) > r >= 0 <= c < len(grid[0]) and grid[r][c] == 1:
            grid[r][c] = 0
            self.dfs(grid, r-1, c)
            self.dfs(grid, r, c-1)
            self.dfs(grid, r+1, c)
            self.dfs(grid, r, c+1)

# 1905. Count Sub Islands
'''
Intuition
You need to know how to count the the number of islands
Refer to 200. Number of Islands

Explanation
Based on solution above,
return 0 if and only if we find it's not a sub island.

Complexity
Time O(mn)
Space O(mn)

 the && operator is short-circuited.
  So, if checkIsland(grid2, grid1, r-1, c) is evaluated to false, 
  then it wouldn't evaluate any of the following checkIsland(grid2, grid1, r+1, c), 
  checkIsland(grid2, grid1, r, c-1) and checkIsland(grid2, grid1, r, c+1). 
  But we don't want that. 
  We want to complete the dfs search for an island in all four directions even if one direction returns false. 
  The &= operator that @lee215 used will not allow for such short-circuiting.
  All the four directions will be explored even after one direction returns false.
'''
class Solution(object):
    def countSubIslands(self, grid1, grid2):
        """
        :type grid1: List[List[int]]
        :type grid2: List[List[int]]
        :rtype: int
        """
        n, m = len(grid2), len(grid2[0])    
        return sum(self.dfs(grid1, grid2, i, j) for i in range(n) for j in xrange(m) if grid2[i][j])
    def dfs(self, grid1, grid2, i, j):
        if not (0 <= i < len(grid2) and 0 <= j < len(grid2[0]) and grid2[i][j] == 1): return 1
        grid2[i][j] = 0
        res = grid1[i][j]
        for di, dj in [[0, 1], [1, 0], [-1, 0], [0, -1]]:
            res &= self.dfs(grid1, grid2, i + di, j + dj)
        return res

# Day 4 Matrix Related Problems
# 1162. As Far from Land as Possible
class Solution(object):
    def maxDistance(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n, steps = len(grid), -1
        dist = [[0] * n for _ in range(n)]
        q = [(i, j) for i in range(n) for j in range(n) if grid[i][j] == 1]
        for i, j in q:
            for r, c in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if r >= 0 and r < n and c >= 0 and c < n and not grid[r][c] and not dist[r][c]:
                    dist[r][c] = dist[i][j] + 1
                    steps = max(steps, dist[r][c])
                    q.append((r, c))
        return steps

# 417. Pacific Atlantic Water Flow
class Solution(object):
    def pacificAtlantic(self, heights):
        """
        :type heights: List[List[int]]
        :rtype: List[List[int]]
        """
        if not heights: return []
        self.directions = [(1,0),(-1,0),(0,1),(0,-1)]
        m = len(heights)
        n = len(heights[0])
        p_visited = [[False for _ in range(n)] for _ in range(m)]
        a_visited = [[False for _ in range(n)] for _ in range(m)]
        result = []
        
        for i in range(m):
            # p_visited[i][0] = True
            # a_visited[i][n-1] = True
            self.dfs(heights,float('-inf'), i, 0, p_visited)
            self.dfs(heights,float('-inf'), i, n-1, a_visited)
        for j in range(n):
            # p_visited[0][j] = True
            # a_visited[m-1][j] = True
            self.dfs(heights,float('-inf'), 0, j, p_visited)
            self.dfs(heights,float('-inf'), m-1, j, a_visited)
            
        for i in range(m):
            for j in range(n):
                if p_visited[i][j] and a_visited[i][j]:
                    result.append([i,j])
        return result
                
    def dfs(self, matrix,height, i, j, visited):
        m, n = len(matrix), len(matrix[0])
        if i < 0 or i >= m or j < 0 or j >= n or visited[i][j] or matrix[i][j] < height: return
        visited[i][j] = True
        for dir in self.directions:
            x, y = i + dir[0], j + dir[1]
            self.dfs(matrix, matrix[i][j], x, y, visited)

# Day 5 Matrix Related Problems
# 1091. Shortest Path in Binary Matrix
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])
        if grid[0][0] == 1 or grid[m-1][n-1] == 1: return -1
        
        q = deque([(0, 0)])  # pair of (r, c)
        dist = 1
        while q:
            for _ in range(len(q)):
                r, c = q.popleft()
                if r == m-1 and c == n-1: return dist
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0: continue
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr == m or nc < 0 or nc == n or grid[nr][nc] == 1: continue
                        grid[nr][nc] = 1  # marked as visited
                        q.append((nr, nc))
            dist += 1
        return -1
# BFS
# we can set the visited grid as non-empty to avoid revisiting.
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        if grid[0][0] or grid[n-1][n-1]:
            return -1
        q = [(0, 0, 1)]
        grid[0][0] = 1
        for i, j, d in q:
            if i == n-1 and j == n-1: return d
            for x, y in ((i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)):
                if 0 <= x < n and 0 <= y < n and not grid[x][y]:
                    grid[x][y] = 1
                    q.append((x, y, d+1))
        return -1

# BFS with deque without corrupting the input:
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        if grid[0][0] or grid[n-1][n-1]:
            return -1
        dirs = [[1,0], [-1,0], [0,1], [0,-1], [-1,-1], [1,1], [1,-1], [-1,1]]
        seen = set()
        queue = collections.deque([(0,0,1)]) # indice, dist
        seen.add((0,0))
        while queue:
            i,j,dist = queue.popleft()
            if i == n -1 and j == n - 1:
                return dist
            for d1, d2 in dirs: 
                x, y = i + d1, j + d2
                if 0 <= x < n and 0 <= y < n:
                    if (x,y) not in seen and grid[x][y] == 0:
                        seen.add((x, y))
                        queue.append((x, y, dist + 1))
        return -1

# 542. 01 Matrix
# BFS
class Solution(object):
    def updateMatrix(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        m, n = len(mat), len(mat[0])
        DIR = [0, 1, 0, -1, 0]

        q = deque([])
        for r in range(m):
            for c in range(n):
                if mat[r][c] == 0:
                    q.append((r, c))
                else:
                    mat[r][c] = -1  # Marked as not processed yet!

        while q:
            r, c = q.popleft()
            for i in range(4):
                nr, nc = r + DIR[i], c + DIR[i + 1]
                if nr < 0 or nr == m or nc < 0 or nc == n or mat[nr][nc] != -1: continue
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
        return mat
# DP
class Solution(object):
    def updateMatrix(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: List[List[int]]
        """
        m = len(mat)
        n = len(mat[0])
        ans = [[ float('inf')-m*n for b in range(n)] for a in range(m)]
        for i in range(0, m):
            for j in range(0, n):
                if mat[i][j]:
                    if i > 0: 
                        ans[i][j] = min(ans[i][j], ans[i - 1][j] + 1)
                    if j > 0: 
                        ans[i][j] = min(ans[i][j], ans[i][j - 1] + 1)
                else :
                    ans[i][j] = 0
            
        for i in range(m - 1, -1,-1):
            for j in range(n - 1, -1,-1):
                if i < m - 1: 
                    ans[i][j] = min(ans[i][j], ans[i + 1][j] + 1)
                if j < n - 1:
                    ans[i][j] = min(ans[i][j], ans[i][j + 1] + 1)
          
        return ans

# Day 6 Matrix Related Problems
# 934. Shortest Bridge
class Solution(object):
    def shortestBridge(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        queue = []
        found = False
        for i in range(len(grid)):
            if found:
                break
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    self.dfs(grid,i,j,queue)
                    found = True
                    break
       
        step = 0
        dirs = [0, 1, 0, -1, 0]
        while queue:
            for i in range(len(queue)):
                temp = queue.pop(0)
                for k in range(4):
                    i = temp[0] + dirs[k]
                    j = temp[1] + dirs[k+1]
                    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j]==2:
                        continue
                    if grid[i][j] == 1:
                        return step
                    grid[i][j] = 2
                    queue.append((i,j))
                    
            step += 1
        return -1
    
    def dfs(self, A,i,j, q):
            if i < 0 or j < 0 or i >= len(A) or j >= len(A[0]) or A[i][j] != 1:
                return
            A[i][j] = 2
            q.append((i,j))
            self.dfs(A,i-1,j,q)
            self.dfs(A,i,j-1,q)
            self.dfs(A,i+1,j,q)
            self.dfs(A,i,j+1,q)
            
# 1926. Nearest Exit from Entrance in Maze
# BFS
class Solution(object):
    def nearestExit(self, maze, entrance):
        """
        :type maze: List[List[str]]
        :type entrance: List[int]
        :rtype: int
        """
        x, y = entrance
        m, n= len(maze), len(maze[0])
        
        q, ans = [], 1
        q.append((x, y))
        maze[x][y] = '+'
        dirs = [-1, 0, 1, 0, -1]
        while q:
            for j in range(len(q)):
                temp = q.pop(0)
                for i in range(1,5):
                    r, c = temp[0] + dirs[i-1], temp[1] + dirs[i]
                    if r>=0 and c>=0 and r<m and c<n and maze[r][c]=='.':
                        if r == 0 or r == m - 1 or c == 0 or c == n - 1:
                            return ans
                        maze[r][c] = '+'
                        q.append((r, c))
            ans+=1
        return -1
# BFS
class Solution(object):
    def nearestExit(self, maze, entrance):
        """
        :type maze: List[List[str]]
        :type entrance: List[int]
        :rtype: int
        """
        x, y = entrance
        m, n, infi = len(maze), len(maze[0]), int(1e5)
        reached = lambda p, q: (not p==x or not q==y) and (p==0 or q==0 or p==m-1 or q==n-1)
        q, ans = deque(), 0
        q.append((x, y, ans))
        directions = [1, 0, -1, 0, 1]
        while q:
            row, col, ans = q.popleft()
            for i in range(4):
                r, c = row+directions[i], col+directions [i+1]
                if r<0 or c<0 or r==m or c==n or maze[r][c]=='+':
                    continue
                if reached(r, c):
                    return ans+1
                maze[r][c] = '+'
                q.append((r, c, ans+1))
        return -1

# Day 7 Standard Traversal
# 797. All Paths From Source to Target
class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        def dfs(cur, path):
            if cur == len(graph) - 1: res.append(path)
            else:
                for i in graph[cur]: dfs(i, path + [i])
        res = []
        dfs(0, [0])
        return res

# 841. Keys and Rooms
# recursive DFS
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        visited = []
        self.dfs(rooms, visited,0);
        return len(visited) == len(rooms)
    
    def dfs(self, rooms, visited, cur=0):
        if cur in set(visited): return
        visited += [cur]
        
        for k in rooms[cur]:
            self.dfs(rooms, visited, k)
# iterative DFS            
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        dfs = [0]
        seen = set(dfs)
        while dfs:
            i = dfs.pop()
            for j in rooms[i]:
                if j not in seen:
                    dfs.append(j)
                    seen.add(j)
                    if len(seen) == len(rooms): return True
        return len(seen) == len(rooms)


# Day 8 Standard Traversal
# 547. Number of Provinces
# DFS
class Solution(object):
    def findCircleNum(self, isConnected):
        """
        :type isConnected: List[List[int]]
        :rtype: int
        """
        def dfs(M, curr, n):
            for i in range(n):
                if M[curr][i] == 1:
                    M[curr][i] = M[i][curr] = 0
                    dfs(M, i, n)
        
        n = len(isConnected)
        ans = 0
        for i in range(n):
            if isConnected[i][i] == 1:
                ans += 1
                dfs(isConnected, i, n)
        
        return ans

# Union Find
class UnionFindSet:
    def __init__(self, n):
        self._parents = [i for i in range(n + 1)]
        self._ranks = [1 for i in range(n + 1)]
    
    def find(self, u):
        while u != self._parents[u]:
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]
        return u
    
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu == pv: return False
        
        if self._ranks[pu] < self._ranks[pv]:
            self._parents[pu] = pv
        elif self._ranks[pu] > self._ranks[pv]:
            self._parents[pv] = pu
        else:        
            self._parents[pv] = pu
            self._ranks[pu] += 1
        
        return True
    
class Solution(object):
    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        
        n = len(M)
        s = UnionFindSet(n)
        for i in range(0,n):
            for j in range(i+1,n):
                if M[i][j] == 1:
                    s.union(i, j)
        
        circles = set()
        for i in range(0,n):
            circles.add(s.find(i))
        return len(circles)

# 1319. Number of Operations to Make Network Connected
# Union Find
# Solution : Union-Find
# Time complexity: O(V+E)
# Space complexity: O(V)
''' Union-Find with Path Compression
Complexity:
Time: O(n+mlogn), m is the length of connections
Space: O(n)
'''
class Solution(object):
    def makeConnected(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        if len(connections) < n - 1: return -1
        import numpy as np
        p = np.arange(0,n)
        cnt = n
        for i, j in connections:
            if self.findParent(p,i) != self.findParent(p,j):
                p[self.findParent(p,i)] = p[self.findParent(p,j)]
                cnt-=1

        return cnt - 1
     
    def findParent(self, parent, i):
        if i == parent[i]: return i
        parent[i] = self.findParent(parent, parent[i])
        return parent[i]
# DFS
'''
Explanation
We need at least n - 1 cables to connect all nodes (like a tree).
If connections.size() < n - 1, we can directly return -1.

One trick is that, if we have enough cables,
we don't need to worry about where we can get the cable from.

We only need to count the number of connected networks.
To connect two unconneccted networks, we need to set one cable.

The number of operations we need = the number of connected networks - 1
'''
# Solution : DFS
# Time complexity: O(V+E)
# Space complexity: O(V+E)
class Solution(object):
    def makeConnected(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        if len(connections) < n - 1: return -1
        G = [set() for i in range(n)]
        for i, j in connections:
            G[i].add(j)
            G[j].add(i)
        seen = [0] * n

        def dfs(i):
            if seen[i]: return 0
            seen[i] = 1
            for j in G[i]: dfs(j)
            return 1

        return sum(dfs(i) for i in range(n)) - 1
# BFS
class Solution(object):
    def makeConnected(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        # if the connections are less than n-1
        # it is impossible to connect all the computers
        if len(connections) < n-1:
            return -1
        
        graph = collections.defaultdict(set)
        for n1, n2 in connections:
            graph[n1].add(n2)
            graph[n2].add(n1)
            
        component = 0 # record how many component we have
        seen = set()
        cnt = {}
        
        for node in graph:
            if node not in seen:
                component += 1
            else:
                continue
            queue = [node]
            seen.add(node)
            for i in queue:
                for nei in graph[i]:
                    if nei not in seen:
                        queue.append(nei)
                        seen.add(nei)
            cnt[node] = len(queue) # this is the size of the connected component

        return n - sum(cnt.values()) + component - 1
            
# Day 9 Standard Traversal
# 1376. Time Needed to Inform All Employees
# Solution 1: Top down DFS
class Solution(object):
    def numOfMinutes(self, n, headID, manager, informTime):
        """
        :type n: int
        :type headID: int
        :type manager: List[int]
        :type informTime: List[int]
        :rtype: int
        """
        children = [[] for i in range(n)]
        for i, m in enumerate(manager):
            if m >= 0: children[m].append(i)

        def dfs(i):
            return max([dfs(j) for j in children[i]] or [0]) + informTime[i]
        return dfs(headID)

# Solution : Bottom Up DFS
class Solution(object):
    def numOfMinutes(self, n, headID, manager, informTime):
        """
        :type n: int
        :type headID: int
        :type manager: List[int]
        :type informTime: List[int]
        :rtype: int
        """
        def dfs(i):
            if manager[i] != -1:
                informTime[i] += dfs(manager[i])
                manager[i] = -1
            return informTime[i]
        return max(map(dfs, range(n)))

# 802. Find Eventual Safe States
# DFS
class Solution(object):
    def eventualSafeNodes(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[int]
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = collections.defaultdict(int)

        def dfs(node):
            if color[node] != WHITE:
                return color[node] == BLACK

            color[node] = GRAY
            for nei in graph[node]:
                if color[nei] == BLACK:
                    continue
                if color[nei] == GRAY or not dfs(nei):
                    return False
            color[node] = BLACK
            return True

        return filter(dfs, range(len(graph)))

# Day 10 Standard Traversal
# 1129. Shortest Path with Alternating Colors
'''
Just need to be noticed that, result can to bigger than n.
To be more specific, the maximum result can be n * 2 - 3.
So in my solution I initial the result as n * 2

Some note:
G = graph
i = index of Node
c = color
'''
class Solution(object):
    def shortestAlternatingPaths(self, n, redEdges, blueEdges):
        """
        :type n: int
        :type redEdges: List[List[int]]
        :type blueEdges: List[List[int]]
        :rtype: List[int]
        """
        G = [[[], []] for i in xrange(n)]
        for i, j in redEdges: G[i][0].append(j)
        for i, j in blueEdges: G[i][1].append(j)
        res = [[0, 0]] + [[n * 2, n * 2] for i in xrange(n - 1)]
        bfs = [[0, 0], [0, 1]]
        for i, c in bfs:
            for j in G[i][c]:
                if res[j][c] == n * 2:
                    res[j][c] = res[i][1 - c] + 1
                    bfs.append([j, 1 - c])
        return [x if x < n * 2 else -1 for x in map(min, res)]

# 1466. Reorder Routes to Make All Paths Lead to the City Zero
'''
Let us put all the edges into adjacency list twice, one with weight 1 and one with weight -1 with oppisite direction. 
Then what we do is just traverse our graph using usual dfs, 
and when we try to visit some neighbour, we check if this edge is usual or reversed.
Complexity is O(V+E), because we traverse our graph only once.
'''
class Solution(object):
    def minReorder(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: int
        """
        self.visited = [0] * n
        self.adj = defaultdict(list) 
        self.count = 0
        for i, j in connections:
            self.adj[i].append([j,1])
            self.adj[j].append([i,-1])

        self.dfs(0)
        return self.count
    
    def dfs(self, start):
        self.visited[start] = 1
        for neib in self.adj[start]:
            if self.visited[neib[0]] == 0:
                if neib[1] == 1:
                    self.count += 1
                self.dfs(neib[0])
# 847. Shortest Path Visiting All Nodes
# Approach : DFS + Memoization (Top-Down DP)
class Solution(object):
    def shortestPathLength(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: int
        """
        def dp(node, mask):
            state = (node, mask)
            if state in cache:
                return cache[state]
            if mask & (mask - 1) == 0:
                # Base case - mask only has a single "1", which means
                # that only one node has been visited (the current node)
                return 0

            cache[state] = float("inf") # Avoid infinite loop in recursion
            for neighbor in graph[node]:
                if mask & (1 << neighbor):
                    already_visited = 1 + dp(neighbor, mask)
                    not_visited = 1 + dp(neighbor, mask ^ (1 << node))
                    cache[state] = min(cache[state], already_visited, not_visited)

            return cache[state]

        n = len(graph)
        ending_mask = (1 << n) - 1
        cache = {}

        return min(dp(node, ending_mask) for node in range(n))
# Approach : BFS
class Solution(object):
    def shortestPathLength(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: int
        """
        if len(graph) == 1:
            return 0
        
        n = len(graph)
        ending_mask = (1 << n) - 1
        queue = [(node, 1 << node) for node in range(n)]
        seen = set(queue)

        steps = 0
        while queue:
            next_queue = []
            for i in range(len(queue)):
                node, mask = queue[i]
                for neighbor in graph[node]:
                    next_mask = mask | (1 << neighbor)
                    if next_mask == ending_mask:
                        return 1 + steps
                    
                    if (neighbor, next_mask) not in seen:
                        seen.add((neighbor, next_mask))
                        next_queue.append((neighbor, next_mask))
            
            steps += 1
            queue = next_queue
# Day 11 Breadth-First Search
# 1306. Jump Game III
'''
Check 0 <= i < A.length
flip the checked number to negative A[i] = -A[i]
If A[i] == 0, get it and return true
Continue to check canReach(A, i + A[i]) and canReach(A, i - A[i])

Complexity
Time O(N), as each number will be flipper at most once.
Space O(N) for recursion.
'''
class Solution(object):
    def canReach(self, arr, start):
        """
        :type arr: List[int]
        :type start: int
        :rtype: bool
        """
        if 0 <= start < len(arr) and arr[start] >= 0:
            arr[start] = -arr[start]
            return arr[start] == 0 or self.canReach(arr, start + arr[start]) or self.canReach(arr, start - arr[start])
        return False

# 1654. Minimum Jumps to Reach Home
# BFS
class Solution(object):
    def minimumJumps(self, forbidden, a, b, x):
        """
        :type forbidden: List[int]
        :type a: int
        :type b: int
        :type x: int
        :rtype: int
        """
        dq, seen, steps, furthest = deque([(True, 0)]), {(True, 0)}, 0, max(x, max(forbidden)) + a + b
        for pos in forbidden:
            seen.add((True, pos)) 
            seen.add((False, pos)) 
        while dq:
            for _ in range(len(dq)):
                dir, pos = dq.popleft()
                if pos == x:
                    return steps
                forward, backward = (True, pos + a), (False, pos - b)
                if pos + a <= furthest and forward not in seen:
                    seen.add(forward)
                    dq.append(forward)
                if dir and pos - b > 0 and backward not in seen:
                    seen.add(backward)
                    dq.append(backward)    
            steps += 1         
        return -1

# 365. Water and Jug Problem
# Math
class Solution(object):
    def canMeasureWater(self,x, y, z):
        """
        :type jug1Capacity: int
        :type jug2Capacity: int
        :type targetCapacity: int
        :rtype: bool
        """
        from fractions import gcd
        return z == 0 or x + y >= z and z % gcd(x, y) == 0

# Day 12 Breadth-First Search
# 433. Minimum Genetic Mutation
# Solution: BFS Shortest Path
# Time complexity: O(n^2)
# Space complexity: O(n)
class Solution(object):
    def minMutation(self, start, end, bank):
        """
        :type start: str
        :type end: str
        :type bank: List[str]
        :rtype: int
        """
        def validMutation(s1, s2):
            count = 0
            for i in range(len(s1)):
                if s1[i] != s2[i]:
                    count += 1
            return count == 1
        
        queue = collections.deque()
        queue.append([start, start, 0]) # current, previous, num_steps
        while queue:
            current, previous, num_steps = queue.popleft()
            if current == end:  # in BFS, the first instance of current == end will yield the minimum
                return num_steps
            for string in bank:
                if validMutation(current, string) and string != previous:
                    queue.append([string, current, num_steps+1])
        return -1
# 752. Open the Lock
class Solution(object):
    def openLock(self, deadends, target):
        """
        :type deadends: List[str]
        :type target: str
        :rtype: int
        """
        from collections import deque
        bases = [1, 10, 100, 1000]
        deads = set(int(x) for x in deadends)
        start, goal = int('0000'), int(target)        
        if start in deads: return -1
        if start == goal: return 0
        q = deque([(start, 0)])
        visited = set([start])
        while q:   
            node, step = q.popleft()
            for i in range(0, 4):
                r = (node // bases[i]) % 10
                for j in [-1, 1]:
                    nxt = node + ((r + j + 10) % 10 - r) * bases[i]
                    if nxt == goal: return step + 1
                    if nxt in deads or nxt in visited: continue
                    q.append((nxt, step + 1))
                    visited.add(nxt)
        return -1

# 127. Word Ladder
# Solution : Bidirectional BFS
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        wordDict = set(wordList)
        if endWord not in wordDict: return 0
        
        l = len(beginWord)
        s1 = {beginWord}
        s2 = {endWord}
        wordDict.remove(endWord)
        step = 0
        while len(s1) > 0 and len(s2) > 0:
            step += 1
            if len(s1) > len(s2): s1, s2 = s2, s1
            s = set()   
            for w in s1:
                new_words = [
                    w[:i] + t + w[i+1:]  for t in string.ascii_lowercase for i in xrange(l)]
                for new_word in new_words:
                    if new_word in s2: return step + 1
                    if new_word not in wordDict: continue
                    wordDict.remove(new_word)                        
                    s.add(new_word)
            s1 = s
        
        return 0
# Day 13 Graph Theory
# 997. Find the Town Judge
'''
Intuition:
Consider trust as a graph, all pairs are directed edge.
The point with in-degree - out-degree = N - 1 become the judge.

Explanation:
Count the degree, and check at the end.

Time Complexity:
Time O(T + N), space O(N)
'''
class Solution(object):
    def findJudge(self, n, trust):
        """
        :type n: int
        :type trust: List[List[int]]
        :rtype: int
        """
        count = [0] * (n + 1)
        for i, j in trust:
            count[i] -= 1
            count[j] += 1
        for i in range(1, n + 1):
            if count[i] == n - 1:
                return i
        return -1

# 1557. Minimum Number of Vertices to Reach All Nodes
'''
Intuition
Just return the nodes with no in-degres.

Explanation
Quick prove:

Necesssary condition: All nodes with no in-degree must in the final result,
because they can not be reached from
All other nodes can be reached from any other nodes.

Sufficient condition: All other nodes can be reached from some other nodes.

Complexity
Time O(E)
Space O(N)
'''
class Solution(object):
    def findSmallestSetOfVertices(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        return list(set(range(n)) - set(j for i, j in edges))

# Day 14 Graph Theory
# 1615. Maximal Network Rank
'''
Build the graph of the roads;
For each pairs of cities, check if they are adjacent and then compute their network rank;
 update the max value of the network rank.
Analysis:
Time & space: O(n ^ 2)
'''
class Solution(object):
    def maximalNetworkRank(self, n, roads):
        """
        :type n: int
        :type roads: List[List[int]]
        :rtype: int
        """
        g = collections.defaultdict(set)
        for a, b in roads:
            g[a].add(b)
            g[b].add(a)
        return max(len(g[a]) + len(g[b]) - (b in g[a])
                    for a in range(n) 
                    for b in range(a + 1, n))

# 886. Possible Bipartition
# BFS
class Solution(object):
    def possibleBipartition(self, n, dislikes):
        """
        :type n: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        colors = [0]*n
        g = collections.defaultdict(set)
        for b in dislikes:
            g[b[0]-1].add(b[1]-1)
            g[b[1]-1].add(b[0]-1)
       
        q = deque()
        for i in range(n):
            if colors[i] != 0: continue
            q.append(i)
            colors[i] = 1
            while q:
                cur = q.popleft()
                for n in g[cur]:
                    if colors[n] == colors[cur]:
                        return False
                    if colors[n] != 0: continue
                    colors[n] = -colors[cur]
                    q.append(n)
        
        return True
        
# DFS
class Solution(object):
    def possibleBipartition(self, n, dislikes):
        """
        :type n: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        colors = [0]*n
        g = collections.defaultdict(set)
        for b in dislikes:
            g[b[0]-1].add(b[1]-1)
            g[b[1]-1].add(b[0]-1)
        def dfs(cur,color):
            colors[cur] = color
            for i in g[cur]:
                if colors[i] == color:
                    return False
                elif colors[i] == 0 and dfs(i, -color) == False:
                    return False
            return True
        for i in range(n):
            if colors[i] == 0 and dfs(i, 1) == False:
                return False
        return True

# 785. Is Graph Bipartite?
# DFS
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        color = {}
        def dfs(pos):
            for i in graph[pos]:
                if i in color:
                    if color[i] == color[pos]:
                        return False
                else:
                    color[i] = 1 - color[pos]
                    if not dfs(i):
                        return False
            return True
        for i in range(len(graph)):
            if i not in color:
                color[i] = 0
                if not dfs(i):
                    return False
        return True

# BFS
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        color = {}
        def bfs(x):
            q = deque([x])
            color[x] = 1
            while q:
                cur = q.popleft()
                for n in graph[cur]:
                    if n not in color:
                        color[n] = -color[cur]
                        q += n,
                    elif color[n] == color[cur]:
                        return False
            return True
        
        return all(i in color or bfs(i) for i in range(len(graph)))

# DFS
class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        color = {}
        def dfs(x, c):
            if x in color: return color[x] == c
            color[x] = c
            return all(dfs(y, -c) for y in graph[x])
        
        return all(i in color or dfs(i, 1) for i in range(len(graph)))