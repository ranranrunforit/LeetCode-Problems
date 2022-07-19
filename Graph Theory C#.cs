// Day 1 Matrix Related Problems
// 733. Flood Fill
public class Solution {
    public int[][] FloodFill(int[][] image, int sr, int sc, int newColor) {
        if (image[sr][sc] != newColor) dfs(image, sr, sc, image[sr][sc], newColor);
        return image;

    }
    
    public void dfs(int[][] image, int sr, int sc, int color, int newColor) {
        if (sr < 0 || sc < 0 || sr >= image.Length || sc >= image[0].Length || image[sr][sc] != color )  return; //image[sr][sc] != color check if the same color
        image[sr][sc] = newColor;
        dfs(image, sr-1, sc, color, newColor);
        dfs(image, sr, sc-1, color, newColor);
        dfs(image, sr+1, sc, color, newColor);
        dfs(image, sr, sc+1, color, newColor);

    }
    
}

// 200. Number of Islands
public class Solution {
    public int NumIslands(char[][] grid) {
        if (grid.Length == 0) return 0;

        int cnt = 0;
        for(int i = 0; i<grid.Length; i++){
            for(int j = 0; j <grid[0].Length; j++){
                if (grid[i][j] == '1'){
                    // Each time when I see a '1', I increment the counter and then erase all connected '1's
                    cnt++;
                    dfs(grid,i,j,grid.Length,grid[0].Length);
                }
            }
        }
        return cnt;
    }
    
    private void dfs(char[][] grid, int x, int y, int n, int m) {
        if (x<0 || y <0 || x>= n || y>= m || grid[x][y] == '0') return;
        grid[x][y] = '0';
        dfs(grid,x-1,y,n,m);
        dfs(grid,x,y-1,n,m);
        dfs(grid,x,y+1,n,m);
        dfs(grid,x+1,y,n,m);
    }
}

// Day 2 Matrix Related Problems
// 695. Max Area of Island
public class Solution {
    public int MaxAreaOfIsland(int[][] grid) {
        int max_area = 0;
        for(int i = 0; i < grid.Length; i++)
            for(int j = 0; j < grid[0].Length; j++)
                if(grid[i][j] == 1)
                    max_area = Math.Max(max_area, AreaOfIsland(grid, i, j));
        return max_area;
    }
    
    public int AreaOfIsland(int[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.Length || j >= grid[0].Length || grid[i][j] == 0) return 0;
        grid[i][j] = 0;
        return 1 + AreaOfIsland(grid, i-1, j) + AreaOfIsland(grid, i, j-1) + AreaOfIsland(grid, i+1, j) + AreaOfIsland(grid, i, j+1);
        
    }
}

// 1254. Number of Closed Islands

/*
Traverse grid, for each 0, do DFS to check if it is a closed island;
Within each DFS, if the current cell is out of the boundary of grid, return 0; 
if the current cell value is positive, return 1; 
otherwise, it is 0, change it to 2 then recurse to its 4 neighors and return the multification of them.
*/
//DFS
public class Solution {
    public int ClosedIsland(int[][] grid) {
        int cnt = 0;
        for (int i = 0; i < grid.Length; ++i)
            for (int j = 0; j < grid[0].Length; ++j)
                if (grid[i][j] == 0)
                    cnt += dfs(i, j, grid);
        return cnt;
    }
    
    private int dfs(int i, int j, int[][] g) {
        if (i < 0 || i >= g.Length || j < 0 || j >= g[0].Length)
            return 0;
        if (g[i][j] > 0)
            return 1;
        g[i][j] = 2; // it is 0
        return dfs(i + 1, j, g) * dfs(i - 1, j, g) * dfs(i, j + 1, g) * dfs(i, j - 1, g);
    }
}
//DFS
public class Solution {
    public int ClosedIsland(int[][] grid) {
        int cnt = 0;
        for (int i = 0; i < grid.Length; ++i)
            for (int j = 0; j < grid[0].Length; ++j)
                if (grid[i][j] == 0)
                    cnt += dfs(i, j, grid);
        return cnt;
    }
    
    private int dfs(int i, int j, int[][] g) {
        if (i < 0 || i >= g.Length || j < 0 || j >= g[0].Length) // wall
            return 0;
        if (g[i][j] > 0) // seen 
            return 1;
        g[i][j] = 1;
        return dfs(i + 1, j, g) + dfs(i - 1, j, g) + dfs(i, j + 1, g) + dfs(i, j - 1, g) == 4 ? 1 : 0;
    }
}
// BFS
//For each land never seen before, BFS to check if the land extends to boundary.
// If yes, return 0, if not, return 1.
// Analysis:
// Time & space: O(m * n), where m = grid.Length, n = grid[0].Length.
public class Solution {
    private static int[] d = {0, 1, 0, -1, 0};
    private int m, n;
    
    public int ClosedIsland(int[][] grid) {
        int cnt = 0; 
        m = grid.Length; n = m == 0 ? 0 : grid[0].Length;
        List<int> seenLand = new List<int>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0 && seenLand.Contains(i * n + j) == false) { // (i, j) is land never seen before.
                    seenLand.Add(i * n + j);
                    cnt += bfs(i, j, grid, seenLand);
                }
            }
        }    
        return cnt;
    }
    
    private int bfs(int i, int j, int[][] g, List<int> seenLand) {
        int ans = 1;
        Queue<int> q = new Queue<int>();
        q.Enqueue(i * n + j);
        while (q.Count!= 0 ) {
            i = q.Peek() / n; j = q.Dequeue() % n;
            for (int k = 0; k < 4; ++k) { // traverse 4 neighbors of (i, j)
                int r = i + d[k], c = j + d[k + 1];
                if (r < 0 || r >= m || c < 0 || c >= n) { // out of boundary.
                    ans = 0; // set 0;
                }else if (g[r][c] == 0 && seenLand.Contains(r * n + c) == false) { // (r, c) is land never seen before.           
                    seenLand.Add(r * n + c);
                    q.Enqueue(r * n + c);
                }
            }
        }
        return ans;
    }
}

// Day 3 Matrix Related Problems
// 1020. Number of Enclaves
/*
Traverse the 4 boundaries and sink all non-enclave lands by DFS, then count the remaining.
Analysis:
Time: O(m * n), space: O(m * n)
*/
public class Solution {
    public int NumEnclaves(int[][] grid) {
        int cnt = 0, m = grid.Length, n = grid[0].Length;
        for (int i = 0; i < m; ++i) { // traverse 4 boundaries and sink lands. 
            int step = (i == 0 || i == m - 1 || n < 3) ? 1 : n - 1; // only in top and bottom rows the step is 1. 
            for (int j = 0; j < n; j += step) {
                if (grid[i][j] > 0) { 
                    dfs(grid, i, j); 
                }
            }
        }
        for (int i = 1; i < m - 1; ++i) { // traverse and count all lands.
            for (int j = 1; j < n - 1; ++j) {
                if (grid[i][j] > 0) 
                    ++cnt;
            }
        }
        return cnt;
    }
    
    public void dfs(int[][] grid, int i, int j) {// sink using DFS.
        if (i < 0 || i >= grid.Length || j < 0 || j >= grid[0].Length || grid[i][j] < 1) { // out of boundary or water.
            return; 
        }
        // else grid[i][j] = 1
        grid[i][j] = -1; // sink the land.
        dfs(grid, i + 1, j);
        dfs(grid, i - 1, j);
        dfs(grid, i, j + 1);
        dfs(grid, i, j - 1);
    }
}
// 1905. Count Sub Islands
/*
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

& <-- verifies both operands
&& <-- stops evaluating if the first operand evaluates to false since the result will be false

(x != 0) & (1/x > 1) <-- this means evaluate (x != 0) then evaluate (1/x > 1) then do the &. 
the problem is that for x=0 this will throw an exception.
(x != 0) && (1/x > 1) <-- this means evaluate (x != 0) and only if this is true then evaluate (1/x > 1) 
so if you have x=0 then this is perfectly safe and won't throw any exception if (x != 0) evaluates to false the whole thing directly evaluates to false without evaluating the (1/x > 1).

EDIT:

exprA | exprB <-- this means evaluate exprA then evaluate exprB then do the |.
exprA || exprB <-- this means evaluate exprA and only if this is false then evaluate exprB and do the ||.
*/
public class Solution {
    public int CountSubIslands(int[][] grid1, int[][] grid2) {
        int m = grid2.Length, n = grid2[0].Length, res = 0;
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (grid2[i][j] == 1)
                    res += dfs(grid1, grid2, i, j);
        return res;
    }


    private int dfs(int[][] grid1, int[][] grid2, int i, int j) {
        int m = grid2.Length, n = grid2[0].Length, res = 1;
        if (i < 0 || i == m || j < 0 || j == n || grid2[i][j] == 0) return 1;
        // This is the condition where we are outside the border of our island and this is where our search ends.
        //If we return 0 from here, then the dfs for all the tiles in the path to the border will always evaluate to zero.
        //If we return 1 from here, then the dfs for a tile will equal to the && of the all the tiles in the path to the border, which is what we want.
        grid2[i][j] = 0; // else grid2[i][j] == 1
        res &= dfs(grid1, grid2, i - 1, j);
        res &= dfs(grid1, grid2, i + 1, j);
        res &= dfs(grid1, grid2, i, j - 1);
        res &= dfs(grid1, grid2, i, j + 1);
        return res & grid1[i][j] ; //grid1[i][j] == 1
    }
}

public class Solution {
    public int CountSubIslands(int[][] grid1, int[][] grid2) {
        int count = 0;
        for(int r=0; r<grid2.Length; r++){
            for(int c=0; c<grid2[0].Length ; c++){
                if(grid2[r][c] == 1 && checkIsland(grid2, grid1, r, c)){
                    count++;
                }
            }
        }
        return count;
}

private bool checkIsland(int[][] grid2, int[][] grid1, int r, int c){
    if(r < 0 || r >= grid2.Length || c < 0 || c >= grid2[0].Length || grid2[r][c] == 0) return true;
    grid2[r][c] = 0;
    
    return checkIsland(grid2, grid1, r-1, c) 
        & checkIsland(grid2, grid1, r+1, c) 
        & checkIsland(grid2, grid1, r, c-1) 
        & checkIsland(grid2, grid1, r, c+1) 
        & grid1[r][c] == 1;
    }
}
// Day 4 Matrix Related Problems
// 1162. As Far from Land as Possible
// BFS
// Use dist to store the distance from land to water;
// Starting from each cell on land, BFS unfound any water cell, dist and grid values of which are both 0's;
// update steps for each water cell.
// Analysis:
// Time compleixty: O(n^2)
// Space complexity: O(n^2)
// Put all land cells into a queue as source nodes and BFS for water cells, 
// the last expanded one will be the farthest.
public class Solution {
    private static int[] d = {0, 1, 0, -1, 0};
    public int MaxDistance(int[][] grid) {
        int R = grid.Length;
        int C = grid[0].Length;
        int steps = -1;
        int[][] dist = new int[R][];
        Queue<int[]> q = new Queue<int[]>();
        for (int r = 0; r < R; ++r) {
            dist[r] = new int[C];
            for (int c = 0;  c < C; ++c) {
                if (grid[r][c] == 1) {
                    q.Enqueue(new int[]{r, c});
                }
            }
        }
        while (q.Count != 0 ) {
            int[] cur = q.Dequeue();
            for (int k = 0; k < 4; ++k) {
                int r = cur[0] + d[k];
                int c = cur[1] + d[k + 1];
                if (r >= 0 && r < R && c >= 0 && c < C && 
                			grid[r][c] == 0 && dist[r][c] == 0) {
                    dist[r][c] = dist[cur[0]][cur[1]] + 1;
                    q.Enqueue(new int[]{r, c});
                    steps = Math.Max(steps, dist[r][c]);
                }
            }
        }
        return steps;
    }
}

// 417. Pacific Atlantic Water Flow
// Start DFS from each boundary.
// Then find common visited node.
public class Solution {
    public IList<IList<int>> PacificAtlantic(int[][] heights) {
       IList<IList<int>> res = new List<IList<int>>();
        if(heights == null || heights.Length == 0 || heights[0].Length == 0){
            return res;
        }
        int n = heights.Length, m = heights[0].Length;
        bool[][] pacific = new bool[n][];
        bool[][] atlantic = new bool[n][];
        for(int i=0; i<n; i++){
            pacific[i] = new bool[m];
            atlantic[i] = new bool[m];
        }
        for(int i=0; i<n; i++){
            dfs(heights, pacific, Int32.MinValue, i, 0);
            dfs(heights, atlantic, Int32.MinValue, i, m-1);
        }
        for(int i=0; i<m; i++){
            dfs(heights, pacific, Int32.MinValue, 0, i);
            dfs(heights, atlantic, Int32.MinValue, n-1, i);
        }
        for (int i = 0; i < n; i++) 
            for (int j = 0; j < m; j++) 
                if (pacific[i][j] == true && atlantic[i][j] == true) 
                    res.Add(new List<int>(){i, j});
        return res;
    }
    
    int[][]dir = new int[][]{ new int[]{0,1}, new int[]{0,-1}, new int[]{1,0}, new int[]{-1,0}};
    
    public void dfs(int[][] matrix, bool[][] visited, int height, int x, int y){
        int n = matrix.Length, m = matrix[0].Length;
        if(x<0 || x>=n || y<0 || y>=m || visited[x][y] == true || matrix[x][y] < height)
            return;
        visited[x][y] = true;
        foreach (int[] d in dir){
            dfs(matrix, visited, matrix[x][y], x+d[0], y+d[1]);
        }
    }
}

// Day 5 Matrix Related Problems
// 1091. Shortest Path in Binary Matrix
// DFS
public class Solution {
    public int ShortestPathBinaryMatrix(int[][] grid) {
        int m = grid.Length, n = grid[0].Length;
        int[][] dist = new int[m][]; // dist[i][j]: distance of the cell (i,j) to (0,0)
        for (int i = 0; i < m; i++) {
             dist[i] = new int[n];
            Array.Fill(dist[i],Int32.MaxValue);
            /*
            for (int j = 0; j < n; j++) {
                dist[i][j] = Int32.MaxValue;
            }*/
        }
        dist[0][0] = 1;
        if (grid[0][0] == 1 || grid[m-1][n-1] == 1)
            return -1;
        grow(grid, dist, 0, 0);
        return (dist[m-1][n-1] != Int32.MaxValue ? dist[m-1][n-1] : -1);
    }
    // Transfer the dist value at (r,c) to or from neighbor cells.
    // Whenever a cell has a updated (smaller) dist value, a recursive call of grow() will be done on behalf of it.
    private void grow(int[][] grid, int[][] dist, int r, int c) {
        int m = grid.Length, n = grid[0].Length;
        int d0 = dist[r][c];
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0)
                    continue;
                int x = r + i;
                int y = c + j;
                if (x >= 0 && x < m && y >= 0 && y < n) {
                    if (grid[x][y] == 1)
                        continue;
                    int d1 = dist[x][y];
                    if (d1 < d0-1) { // get a smaller value from a neighbor; then re-start the process.
                        dist[r][c] = d1+1;
                        grow(grid, dist, r, c); // TODO some optimization to avoid stack overflow
                        return;
                    } else if (d1 > d0+1) { // give a smaller value to a neighbor
                        dist[x][y] = d0+1;
                        grow(grid, dist, x, y);
                    }
                }
            }
        }
    }
}
// BFS
public class Solution {
    public int ShortestPathBinaryMatrix(int[][] grid) {
       int[][] dirs = new int[][]{new int[] {0,1},new int[] {1,0}, new int[] {-1,0}, new int[] {0,-1}, new int[] {1,1}, new int[] {-1,1}, new int[] {1,-1}, new int[] {-1,-1}};
        
        if(grid == null || grid.Length == 0) return -1;
        
        if(grid[0][0] != 0) return -1;
        
        Queue<int[]> queue = new Queue<int[]>();
        queue.Enqueue(new int[] {0,0,0});
        
        while(queue.Count != 0){
            
            int[] curr = queue.Dequeue();
            if(curr[0] == grid.Length-1 && curr[1] == grid[0].Length-1) return curr[2]+1;
            
            
            for(int i=0; i < dirs.Length; i++){
                int x = curr[0] + dirs[i][0];
                int y = curr[1] + dirs[i][1];
                
                if(x < 0 || y < 0 || x >= grid.Length || y >= grid[0].Length || grid[x][y] != 0) continue;
                
                grid[x][y] = -1;
                queue.Enqueue(new int[]{x,y, curr[2]+1});
            }
        }
        
        return -1;
    }
}

// 542. 01 Matrix
// BFS
public class Solution {
    int[] DIR = new int[]{0, 1, 0, -1, 0};
    public int[][] UpdateMatrix(int[][] mat) {
        int m = mat.Length, n = mat[0].Length; // The distance of cells is up to (M+N)
        Queue<int[]> q = new Queue<int[]>();
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < n; ++c)
                if (mat[r][c] == 0) q.Enqueue(new int[]{r, c});
                else mat[r][c] = -1; // Marked as not processed yet!

        while (q.Count != 0) {
            int[] curr = q.Dequeue();
            int r = curr[0], c = curr[1];
            for (int i = 0; i < 4; ++i) {
                int nr = r + DIR[i], nc = c + DIR[i+1];
                if (nr < 0 || nr == m || nc < 0 || nc == n || mat[nr][nc] != -1) continue;
                mat[nr][nc] = mat[r][c] + 1;
                q.Enqueue(new int[]{nr, nc});
            }
        }
        return mat;
    }
}
// DP
public class Solution {
    public int[][] UpdateMatrix(int[][] mat) {
        int m = mat.Length,n = mat[0].Length;
        if (mat.Length == 0) return mat;
        int[][] ans = new int[m][];
        //First pass: check for left and top
        for (int i = 0; i < m; i++){
            ans[i] = new int[n];
            for (int j = 0; j < n; j++){
                ans[i][j] = Int32.MaxValue-m*n;
                if (mat[i][j] == 1 ) 
                { 
                    if (i > 0) {
                        ans[i][j] = Math.Min(ans[i][j], ans[i - 1][j] + 1);
                    }
                    if (j > 0) {
                        ans[i][j] = Math.Min(ans[i][j], ans[i][j - 1] + 1);
                    }                    
                }   
                else 
                {
                    ans[i][j] = 0;
                }
            }
        }
         
         //Second pass: check for bottom and right
        for (int i = m - 1; i >= 0; i--){
            for (int j = n - 1; j >= 0; j--) {
                if (i < m - 1) 
                {
                    ans[i][j] = Math.Min(ans[i][j], ans[i + 1][j] + 1);
                }
                if (j < n - 1) 
                {
                    ans[i][j] = Math.Min(ans[i][j], ans[i][j + 1] + 1);
                }
          }
        }
          
        return ans;
    }
}

// Day 6 Matrix Related Problems
// 934. Shortest Bridge
/*
Solution: DFS + BFS
Use DFS to find one island and color all the nodes as 2 (BLUE).
Use BFS to find the shortest path from any nodes with color 2 (BLUE) to any nodes with color 1 (RED).
Time complexity: O(mn)

Space complexity: O(mn)
*/
public class Solution {
    public int ShortestBridge(int[][] grid) {
        Queue<KeyValuePair<int, int>> q = new Queue<KeyValuePair<int, int>>();
        bool found = false;
        for (int i = 0; i < grid.Length && !found; ++i)
          for (int j = 0; j < grid[0].Length && !found; ++j)
            if (grid[i][j] == 1) {
              dfs(grid, j, i, q);
              found = true;
            }
    
        int steps = 0;
        int[] dirs = new int[]{0, 1, 0, -1, 0};
        while (q.Count != 0) {      
          for ( int size = q.Count; size > 0; size--) {
            int x = q.Peek().Key;
            int y = q.Peek().Value;
            q.Dequeue();
            for (int i = 0; i < 4; ++i) {
              int tx = x + dirs[i];
              int ty = y + dirs[i + 1];
              if (tx < 0 || ty < 0 || tx >= grid[0].Length || ty >= grid.Length || grid[ty][tx] == 2) continue;          
              if (grid[ty][tx] == 1) return steps;
              grid[ty][tx] = 2;
              q.Enqueue(new KeyValuePair<int,int>(tx, ty));
            }
          }
          ++steps;
        }
        return -1;
    }
    
    private void dfs(int[][] A, int x, int y, Queue<KeyValuePair<int, int>> q) {
    if (x < 0 || y < 0 || x >= A[0].Length || y >= A.Length || A[y][x] != 1) return;
    A[y][x] = 2;
    q.Enqueue(new KeyValuePair<int, int>(x, y));
    dfs(A, x - 1, y, q);
    dfs(A, x, y - 1, q);
    dfs(A, x + 1, y, q);
    dfs(A, x, y + 1, q);
  }
}
// 1926. Nearest Exit from Entrance in Maze
// BFS
public class Solution {
    public int NearestExit(char[][] maze, int[] entrance) {
        Queue<int[]> q = new Queue<int[]>();
        int r = 1; int[] dirs = new int[]{-1, 0, 1, 0, -1}; 
        int n = maze.Length, m = maze[0].Length;
        q.Enqueue(entrance);
        maze[entrance[0]][entrance[1]] = '+';
        while (q.Count != 0) {
            for (int size = q.Count; size > 0; size--){
                int[] p = q.Dequeue();
                for (int d = 1; d < dirs.Length; d++) {
                    int x = p[0] + dirs[d - 1], y = p[1] + dirs[d];
                    if (0 <= x && x < n && 0 <= y && y < m && maze[x][y] == '.') {
                        if (x == 0 || x == n - 1 || y == 0 || y == m - 1)
                            return r;
                        maze[x][y] = '+';
                        q.Enqueue(new int[]{x, y});
                    }
                }
            }
            r++;
        }
        return -1;
    }
}

// Day 7 Standard Traversal
// 797. All Paths From Source to Target
/*
If it asks just the number of paths, generally we can solve it in two ways.

Count from start to target in topological order.
Count by dfs with memo.
Both of them have time O(Edges) and O(Nodes) space. Let me know if you agree here.
I didn't do that in this problem, for the reason that it asks all paths. I don't expect memo to save much time. (I didn't test).
Imagine the worst case that we have node-1 to node-N, and node-i linked to node-j if i < j.
There are 2^(N-2) paths and (N+2)*2^(N-3) nodes in all paths. We can roughly say O(2^N).
*/
public class Solution {
    public IList<IList<int>> AllPathsSourceTarget(int[][] graph) {
        IList<IList<int>> paths = new List<IList<int>>();
        IList<int> path = new List<int>();
        dfs(graph, paths, path, 0);
        return paths;
    }
    public void dfs(int[][] g, IList<IList<int>> res, IList<int> path, int cur) {
        path.Add(cur);
        if (cur == g.Length - 1){
            res.Add(new List<int>(path));
        }
        else{
            foreach (int i in g[cur]){
            dfs(g, res, path, i);
            }
        } 
            
        path.RemoveAt(path.Count-1);

    }
}
 
// 841. Keys and Rooms
// recursive DFS
public class Solution {
    public bool CanVisitAllRooms(IList<IList<int>> rooms) {
      List<int> visited = new List<int>();
    dfs(rooms, 0, visited);
    return visited.Count == rooms.Count;
    }
    public void dfs(IList<IList<int>> rooms, 
            int cur, List<int> visited) {
        if (visited.Contains(cur)) return;
        visited.Add(cur);
        foreach (int nxt in rooms[cur])
        dfs(rooms, nxt, visited);
    }
}
// iterative DFS
public class Solution {
    public bool CanVisitAllRooms(IList<IList<int>> rooms) {
        Stack<int> dfs = new Stack<int>(); dfs.Push(0);
            List<int> seen = new List<int>(); seen.Add(0);
            while (dfs.Count != 0) {
                int i = dfs.Pop();
                foreach (int j in rooms[i])
                    if (!seen.Contains(j)) {
                        dfs.Push(j);
                        seen.Add(j);
                        if (rooms.Count == seen.Count) return true;
                    }
            }
            return rooms.Count == seen.Count;
    }
}

// Day 8 Standard Traversal
// 547. Number of Provinces
// Union Find
public class Solution {
    class UF {
        private int[] parent, size;
        private int cnt;

        public UF(int n) {
            parent = new int[n];
            size = new int[n];
            cnt = n;
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 1;
            }

        }

        public int find(int p) {
            // path compression
            while (p != parent[p]) {
                parent[p] = parent[parent[p]];
                p = parent[p];
            }
            return p;
        }

        public void union(int p, int q) {
            int rootP = find(p);
            int rootQ = find(q);
            if (rootP == rootQ) 
                return;
            // union by size
            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP;
                size[rootP] += size[rootQ];
            } else {
                parent[rootP] = rootQ;
                size[rootQ] += size[rootP];
            }
            cnt--;
        }

        public int count() { return cnt; }
}
    
    
    public int FindCircleNum(int[][] isConnected) {
        int n = isConnected.Length;
        UF uf = new UF(n);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                if (isConnected[i][j] == 1)
                    uf.union(i, j);
        return uf.count();
    }

}
// DFS
public class Solution {
    public int FindCircleNum(int[][] isConnected) {
        int n = isConnected.Length;
        if (n == 0) return 0;
        
        int ans = 0;
        for (int i = 0; i < n; ++i) {
            if (isConnected[i][i] == 0) continue;            
            ++ans;
            dfs(isConnected, i, n);
        }
        return ans;
    }
    
    public void dfs(int[][] M, int curr, int n) {
        for (int i = 0; i < n; ++i) {
            if (M[curr][i] == 0) continue;
            M[curr][i] = M[i][curr] = 0;
            dfs(M, i, n);
        }
    }
}

// 1319. Number of Operations to Make Network Connected
// Union Find
// Solution : Union-Find
// Time complexity: O(V+E)
// Space complexity: O(V)
/* Union-Find with Path Compression
Complexity:
Time: O(n+mlogn), m is the length of connections
Space: O(n)
*/
public class Solution {
    public int MakeConnected(int n, int[][] connections) {
        if (connections.Length < n - 1) return -1; // To connect all nodes need at least n-1 edges
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        int components = n;
        foreach (int[] c in connections) {
            int p1 = findParent(parent, c[0]);
            int p2 = findParent(parent, c[1]);
            if (p1 != p2) {
                parent[p1] = p2; // Union 2 component
                components--;
            }
        }
        return components - 1; // Need (components-1) cables to connect components together
    }
    private int findParent(int[] parent, int i) {
        if (i == parent[i]) return i;
        return parent[i] = findParent(parent, parent[i]); // Path compression
    }
    
}

// DFS
// Solution : DFS
// Time complexity: O(V+E)
// Space complexity: O(V+E)
public class Solution {
    public int MakeConnected(int n, int[][] connections) {
        if (connections.Length < n - 1) return -1; // To connect all nodes need at least n-1 edges
        List<int>[] graph = new List<int>[n];
        for (int i = 0; i < n; i++) graph[i] = new List<int>();
        foreach (int[] c in connections) {
            graph[c[0]].Add(c[1]);
            graph[c[1]].Add(c[0]);
        }
        int components = 0;
        bool[] visited = new bool[n];
        for (int v = 0; v < n; v++) components += dfs(v, graph, visited);
        return components - 1; // Need (components-1) cables to connect components together
    }
    int dfs(int u, List<int>[] graph, bool[] visited) {
        if (visited[u]) return 0;
        visited[u] = true;
        foreach (int v in graph[u]) dfs(v, graph, visited);
        return 1;
    }
}

// BFS
// Complexity:
// Time: O(n+m), m is the length of connections
// Space: O(n)
public class Solution {
    public int MakeConnected(int n, int[][] connections) {
        if (connections.Length < n - 1)
        {
            return -1;
        }
        
        var graph = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            graph.Add(i, new List<int>());
        }
        foreach (var connection in connections)
        {
            graph[connection[0]].Add(connection[1]);
            graph[connection[1]].Add(connection[0]);
        }
        
        var queue = new Queue<int>();
        bool[] visited = new bool[n];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (visited[i])
            {
                continue;
            }
            
            queue.Enqueue(i);
            visited[i] = true;
            while (queue.Count > 0)
            {
                int node = queue.Dequeue();
                foreach (var next in graph[node])
                {
                    if (visited[next])
                    {
                        continue;
                    }
                    visited[next] = true;
                    queue.Enqueue(next);
                }
            }
            count++;
        }
        return count - 1;
    }
}
// Day 9 Standard Traversal
// 1376. Time Needed to Inform All Employees
// Solution 1: Top down DFS
// dfs find out the time needed for each employees.
// The time for a manager = max(manager's employees) + informTime[manager]
// Time O(N), Space O(N)
public class Solution {
    public int NumOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        Dictionary<int, List<int>> graph = new Dictionary<int, List<int>>();
        int total = 0;
        for (int i = 0; i < manager.Length; i++) {
            int j = manager[i];
            if (!graph.ContainsKey(j))
                graph[j]= new List<int>();
            graph[j].Add(i);
        }
        return dfs(graph, informTime, headID);
    }

    private int dfs(Dictionary<int, List<int>> graph, int[] informTime, int cur) {
        int max = 0;
        if (!graph.ContainsKey(cur))
            return max;
        for (int i = 0; i < graph[cur].Count; i++)
            max = Math.Max(max, dfs(graph, informTime, graph[cur][i]));
        return max + informTime[cur];
    
    }
}

// Solution 2: Bottom Up DFS
// When you call dfs on a node it asks for information from its manager and the time taken is added to this node's inform time.
// Also mark this node's manager as -1.
// Why? Because you don't want to ask for information again once the information has reached you.
public class Solution {
    public int NumOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        int res = 0;
        for (int i = 0; i < n; ++i)
            res = Math.Max(res, dfs(i, manager, informTime));
        return res;
    }
    public int dfs(int i, int[] manager, int[] informTime) {
        if (manager[i] != -1) {
            informTime[i] += dfs(manager[i], manager, informTime);
            manager[i] = -1;
        }
        return informTime[i];
    
    }
}
// BFS
// Complexity:
// Time & Space: O(N)
public class Solution {
    public int NumOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        List<int>[] graph = new List<int>[n];
        for (int i = 0; i < n; i++) graph[i] = new List<int>();
        for (int i = 0; i < n; i++) if (manager[i] != -1) graph[manager[i]].Add(i);
        Queue<int[]> q = new Queue<int[]>(); // Since it's a tree, we don't need `visited` array
        q.Enqueue(new int[]{headID, 0});
        int ans = 0;
        while (q.Count != 0 ) {
            int[] top = q.Dequeue();
            int u = top[0], w = top[1];
            ans = Math.Max(w, ans);
            foreach (int v in graph[u]) q.Enqueue(new int[]{v, w + informTime[u]});
        }
        return ans;
    
    }
}

// 802. Find Eventual Safe States
// DFS
public class Solution {
    public IList<int> EventualSafeNodes(int[][] graph) {
        int N = graph.Length;
        int[] color = new int[N];
        List<int> ans = new List<int>();

        for (int i = 0; i < N; ++i)
            if (dfs(i, color, graph))
                ans.Add(i);
        return ans;
    }

    // colors: WHITE 0, GRAY 1, BLACK 2;
    public bool dfs(int node, int[] color, int[][] graph) {
        if (color[node] > 0)
            return color[node] == 2;

        color[node] = 1;
        foreach (int nei in graph[node]) {
            if (color[node] == 2)
                continue;
            if (color[nei] == 1 || !dfs(nei, color, graph))
                return false;
        }

        color[node] = 2;
        return true;
    }
}

// Day 10 Standard Traversal
// 1129. Shortest Path with Alternating Colors
/*
Just need to be noticed that, result can to bigger than n.
To be more specific, the maximum result can be n * 2 - 3.
So in my solution I initial the result as n * 2

Some note:
G = graph
i = index of Node
c = color
*/
public class Solution {
    public int[] ShortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        // Two sets one for blu and another for red
        List<int>[][] graph = new List<int>[2][];
        graph[0] = new List<int>[n];
        graph[1] = new List<int>[n];
        for (int i = 0; i < n; i++) {
            graph[0][i] = new List<int>();
            graph[1][i] = new List<int>();
        }
        // red edges in 0 - col
        foreach (int[] re in redEdges) {
            graph[0][ re[0] ].Add(re[1]);
        }
        // blu edges in 1 - col
        foreach (int[] blu in blueEdges) {
            graph[1][ blu[0] ].Add(blu[1]);
        }
        int[][] res = new int[2][];
        res[0] = new int[n];
        res[1] = new int[n];
        // Zero edge is always accessible to itself - leave it as 0
        for (int i = 1; i < n; i++) {
            res[0][i] = 2 * n;
            res[1][i] = 2 * n;
        }
        // Q entries are vert with a color (up to that point)
        Queue<int[]> q = new Queue<int[]>();
        q.Enqueue(new int[] {0, 0}); // either with red
        q.Enqueue(new int[] {0, 1}); // or with blue
        while (q.Count != 0) {
            int[] cur = q.Dequeue();
            int vert = cur[0];
            int colr = cur[1];
            // No need to keep track of level up to now
            // only need to keep what color - and the length
            // is automatically derived from previous node
            foreach (int nxt in graph[1 - colr][vert]) {
                if (res[1 - colr][nxt] == 2 * n) {
                    res[1 - colr][nxt] = 1 + res[colr][vert];
                    q.Enqueue(new int[] {nxt, 1 - colr});
                }
            }
        }
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            int t = Math.Min(res[0][i], res[1][i]);
            ans[i] = (t == 2 * n) ? -1 : t;
        }
        return ans;
    }
}
// 1466. Reorder Routes to Make All Paths Lead to the City Zero
/*
Based on the problem description, we have a tree, and node zero is the root.

However, the direction can point either from a parent to a child (positive), or from a child to its parent (negative). 
To solve the problem, we traverse the tree and count edges that are directed from a parent to a child. 
Direction of those edges need to be changed to arrive at zero node.

In the code below, I am using the adjacency list, and the sign indicates the direction. 
If the index is positive - the direction is from a parent to a child and we need to change it (change += (to > 0)).

Note that we cannot detect the direction for zero (-0 == 0), but it does not matter as we start our traversal from zero.

Complexity Analysis
Time: O(n). We visit each node once.
Memory: O(n). We store n nodes in the adjacency list, with n - 1 edges in total.
*/
public class Solution {
    int dfs(List<List<int>> al, bool[] visited, int f) {
    int change = 0;
    visited[f] = true;
    foreach (var t in al[f])
        if (visited[Math.Abs(t)]!= true)
            change += dfs(al, visited, Math.Abs(t)) + (t > 0 ? 1 : 0);
    return change;   
}
    public int MinReorder(int n, int[][] connections) {
        List<List<int>> al = new List<List<int>>();
        for(int i = 0; i < n; ++i) 
            al.Add(new List<int>());
        foreach (var c in connections) {
            al[c[0]].Add(c[1]);
            al[c[1]].Add(-c[0]);
        }
        return dfs(al, new bool[n], 0);
    }
}
// Bonus: Minimalizm Version
// Instead of visited, we can just pass the previous index to prevent going back, as suggested by zed_b.
// This is possible because every node has only one parent in the tree.
public class Solution {
    int dfs(List<List<int>> al,  int prev, int node) {
        int change = 0;
        foreach (var t in al[node])
            if (Math.Abs(t) != prev)
                change += dfs(al, node, Math.Abs(t)) + (t > 0 ? 1 : 0);
        return change;   
    
}
    public int MinReorder(int n, int[][] connections) {
        List<List<int>> al = new List<List<int>>();
        for(int i = 0; i < n; ++i) 
            al.Add(new List<int>());
        foreach (var c in connections) {
            al[c[0]].Add(c[1]);
            al[c[1]].Add(-c[0]);
        }
        return dfs(al, 0, 0);
    }
}

// 847. Shortest Path Visiting All Nodes
// Approach : DFS + Memoization (Top-Down DP)
public class Solution {
    private int[][] cache;
    private int endingMask;
    public int ShortestPathLength(int[][] graph) {
        int n = graph.Length;
        endingMask = (1 << n) - 1;
        cache = new int[n + 1][];
        for(int i = 0; i < n+1; i++){
            cache[i] = new int[endingMask + 1];
        }
        
        int best = Int32.MaxValue;
        for (int node = 0; node < n; node++) {
            best = Math.Min(best, dp(node, endingMask, graph));
        }
        
        return best;
    }
    
    public int dp(int node, int mask, int[][] graph) {
        if (cache[node][mask] != 0) {
            return cache[node][mask];
        }
        if ((mask & (mask - 1)) == 0) {
            // Base case - mask only has a single "1", which means
            // that only one node has been visited (the current node)
            return 0;
        }
        
        cache[node][mask] = Int32.MaxValue - 1; // Avoid infinite loop in recursion
        foreach (int neighbor in graph[node]) {
            if ((mask & (1 << neighbor)) != 0) {
                int alreadyVisited = dp(neighbor, mask, graph);
                int notVisited = dp(neighbor, mask ^ (1 << node), graph);
                int betterOption = Math.Min(alreadyVisited, notVisited);
                cache[node][mask] = Math.Min(cache[node][mask], 1 + betterOption);
            }
        }
        
        return cache[node][mask];
    }
}

// Approach : BFS
public class Solution {
    public int ShortestPathLength(int[][] graph) {
        if (graph.Length == 1) {
            return 0;
        }
        
        int n = graph.Length;
        int endingMask = (1 << n) - 1;
        bool[][] seen = new bool[n][];
        List<int[]> queue = new List<int[]>();
        
        for (int i = 0; i < n; i++) {
            seen[i] = new bool[endingMask];
            queue.Add(new int[] {i, 1 << i});
            seen[i][1 << i] = true;
        }
        
        int steps = 0;
        while (queue.Count != 0) {
            List<int[]> nextQueue = new List<int[]>();
            for (int i = 0; i < queue.Count; i++) {
                int[] currentPair = queue[i];
                int node = currentPair[0];
                int mask = currentPair[1];
                foreach (int neighbor in graph[node]) {
                    int nextMask = mask | (1 << neighbor);
                    if (nextMask == endingMask) {
                        return 1 + steps;
                    }
                    
                    if (seen[neighbor][nextMask] != true) {
                        seen[neighbor][nextMask] = true;
                        nextQueue.Add(new int[] {neighbor, nextMask});
                    }
                }
            }
            steps++;
            queue = nextQueue;
        }
        
        return -1;
    }
}

// Day 11 Breadth-First Search
// 1306. Jump Game III
/*
Check 0 <= i < A.length
flip the checked number to negative A[i] = -A[i]
If A[i] == 0, get it and return true
Continue to check canReach(A, i + A[i]) and canReach(A, i - A[i])

Complexity
Time O(N), as each number will be flipper at most once.
Space O(N) for recursion.
*/
public class Solution {
    public bool CanReach(int[] arr, int start) {
        return 0 <= start && start < arr.Length && arr[start] >= 0 && ((arr[start] = -arr[start]) == 0 || CanReach(arr, start + arr[start]) || CanReach(arr, start - arr[start]));
    }
}
// 1654. Minimum Jumps to Reach Home
// BFS
public class Solution {
    public int MinimumJumps(int[] forbidden, int a, int b, int x) {
        PriorityQueue<int[],int[]> pq = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a1,a2)=> a1[0] - a2[0]));
        pq.Enqueue(new int[]{0,0,0},new int[]{0,0,0});//step, current index, direction(0 is back, 1 is forward)
        List<int> forbit = new List<int>();
        List<string> visited = new List<string>();
        int maxLimit = 2000 + 2 * b;
        foreach(int num in forbidden){
            forbit.Add(num);
            maxLimit = Math.Max(maxLimit, num + 2 * b);
        }
        while(pq.Count!= 0){
            int[] node = pq.Dequeue();
            int step = node[0];
            int idx = node[1];
            int dir = node[2];
            if(idx == x) return step;
			//try jump forward
            if(idx+a < maxLimit && !forbit.Contains(idx+a) && !visited.Contains(idx+a+","+0)){
                visited.Add(idx+a+","+0);
                pq.Enqueue(new int[]{step+1, idx+a, 0},new int[]{step+1, idx+a, 0});
            }
			//try jump back
            if(idx-b >= 0 && !forbit.Contains(idx-b) && !visited.Contains(idx-b+","+1) && dir != 1){
                visited.Add(idx-b+","+1);
                pq.Enqueue(new int[]{step+1, idx-b, 1},new int[]{step+1, idx-b, 1});
            }
        }
        return -1;         
    }
}
// DFS
public class Solution {
    private Dictionary<String/*idx + direction*/, int> cache;
    public int MinimumJumps(int[] forbidden, int a, int b, int x) {
        cache = new Dictionary<String,int>();
        List<int> visited = new List<int>(); 
        List<int> forbit = new List<int>();
        int maxLimit = 2000 + 2 * b;
        foreach(int num in forbidden){
            forbit.Add(num);
            maxLimit = Math.Max(maxLimit, num + 2 * b);
        }
        int val = dfs(0, x, a, b, forbit, visited, 0, maxLimit);
        return val == Int32.MaxValue ? -1 : val;
    }
    private int dfs(int idx, int x, int a, int b, List<int> forbit, List<int> visited, int dir, int maxLimit){
        if(cache.ContainsKey(idx+","+dir)){
            return cache[idx+","+dir];
        }
        if(idx == x) return 0;
        if(idx < 0 || idx > maxLimit) return Int32.MaxValue;
        visited.Add(idx);
        int min = Int32.MaxValue;
		//try jump forward
        if(idx+a < maxLimit && !forbit.Contains(idx+a) && !visited.Contains(idx+a)){
            int step = dfs(idx+a, x, a, b, forbit, visited, 0, maxLimit);
            if(step != Int32.MaxValue){
                min = Math.Min(min, step + 1);
            }
        }
		//try jump back
       if(idx-b >= 0 && !forbit.Contains(idx-b) && !visited.Contains(idx-b) && dir != 1){
            int step = dfs(idx-b, x, a, b, forbit, visited, 1, maxLimit);
            if(step != Int32.MaxValue){
                min = Math.Min(min, step + 1);
            }
        }
        visited.Remove(idx);
        cache[idx+","+dir] = min;
        return min;
    }
}
// DFS
public class Solution {
    int minJumps = int.MaxValue;
    int max_val;
    int p;
    int q;
    int valueToReach;
    List<int> visited;
    public int MinimumJumps(int[] forbidden, int a, int b, int x) {
        visited = new List<int>();
        for(int i =0;i<forbidden.Length;i++){
            visited.Add(forbidden[i]);
        }
        max_val = 6000;
        p = a;
        q = b;
        valueToReach = x;
        visited.Add(0);
        dfs(0, 0, "forward");
        if(minJumps == Int32.MaxValue)
            return -1;
        return minJumps;
    }
    
    public void dfs(int currNode, int jumps, String previousStep){
        if(currNode == valueToReach){
            minJumps = minJumps < jumps ? minJumps : jumps;
            return;
        }
        if(previousStep.Equals("forward"))
            visited.Add(currNode);
        if(((currNode + p) < max_val) && !visited.Contains(currNode + p)){
            
                dfs(currNode + p, jumps +1, "forward");
        }
        if(((currNode - q) < max_val)&& previousStep.Equals("forward")&& (currNode - q) >=0 && !visited.Contains(currNode - q)){
                
                dfs(currNode - q, jumps + 1, "back");
        }
        return;
    }
}

// 365. Water and Jug Problem
// Math
public class Solution {
    public bool CanMeasureWater(int x, int y, int z) {
        //limit brought by the statement that water is finallly in one or both buckets
        if(x + y < z) return false;
        //case x or y is zero
        if( x == z || y == z || x + y == z ) return true;
        
        //get GCD, then we can use the property of Bézout's identity
        return z%GCD(x, y) == 0;
}

public int GCD(int a, int b){
        while(b != 0 ){
            int temp = b;
            b = a%b;
            a = temp;
        }
        return a;

    }
}

public class Solution {
    public bool CanMeasureWater(int x, int y, int z) {
        if(x + y < z) return false;
        if( x == z || y == z || x + y == z ) return true; 
        return z%gcd(x, y) == 0;
    } 
    
    int gcd(int a, int b){
        if(b == 0) return a;
        return gcd(b, a % b);
    }
}

// Day 12 Breadth-First Search
// 433. Minimum Genetic Mutation
// Solution: BFS Shortest Path
// Time complexity: O(n^2)
// Space complexity: O(n)
public class Solution {
    public int MinMutation(string start, string end, string[] bank) {
        Queue<string> q = new Queue<string>();
    q.Enqueue(start);
    
    List<string> visited = new List<string>();
    visited.Add(start);
    
    int mutations = 0;
    while (q.Count != 0) {
      int size = q.Count;
      for(int i = size; i>0; i--) {
        string curr = q.Peek(); q.Dequeue();
        if (curr == end) return mutations;
        foreach (string gene in bank) {
          if (visited.Contains(gene) || !validMutation(curr, gene)) continue;
          visited.Add(gene);
          q.Enqueue(gene);
        }
      }
      ++mutations;
    }    
    return -1;
  }
private bool validMutation(string s1, string s2) {
    int count = 0;
    for (int i = 0; i < s1.Length; ++i)
      if (s1[i] != s2[i] && count++ > 0) return false;
    return true;
  
    }
}
// 752. Open the Lock
// BFS TLE
public class Solution {
    public int OpenLock(string[] deadends, string target) {
        String start = "0000";
        List<string> dead = new List<string>();
        foreach (String d in deadends) dead.Add(d);
        if (dead.Contains(start)) return -1;

        Queue<String> queue = new Queue<String>();
        queue.Enqueue(start);        

        List<string> visited = new List<string>();
        visited.Add(start);

        int steps = 0;
        while (queue.Count != 0) {
          ++steps;
          int size = queue.Count;
          for (int s = 0; s < size; ++s) {
            String node = queue.Peek();
            for (int i = 0; i < 4; ++i) {
              for (int j = -1; j <= 1; j += 2) {
                char[] chars = node.ToCharArray();
                chars[i] = (char)(((chars[i] - '0') + j + 10) % 10 + '0');
                String next = new String(chars);
                if (next.Equals(target)) return steps;
                if (dead.Contains(next) || visited.Contains(next))
                    continue;
                visited.Add(next);
                queue.Enqueue(next);
              }
            }
          }
        }
        return -1;
    }
}

// 127. Word Ladder
// Solution : Bidirectional BFS
public class Solution {
    public int LadderLength(string beginWord, string endWord, IList<string> wordList) {
        List<string> dict = new List<string>();
    foreach (String word in wordList) dict.Add(word);
    
    if (!dict.Contains(endWord)) return 0;
    
    List<string> q1 = new List<string>();
    List<string> q2 = new List<string>();
    q1.Add(beginWord);
    q2.Add(endWord);
    
    int l = beginWord.Length;
    int steps = 0;
    
    while (q1.Count != 0 && q2.Count != 0) {
      ++steps;
      
      if (q1.Count > q2.Count) {
        List<String> tmp = q1;
        q1 = q2;
        q2 = tmp;
      }
      
      List<String> q = new List<String>();
      
      foreach (String w in q1) {        
        char[] chs = w.ToCharArray();
        for (int i = 0; i < l; ++i) {
          char ch = chs[i];
          for (char c = 'a'; c <= 'z'; ++c) {
            chs[i] = c;
            String t = new String(chs);         
            if (q2.Contains(t)) return steps + 1;            
            if (!dict.Contains(t)) continue;            
            dict.Remove(t);        
            q.Add(t);
          }
          chs[i] = ch;
        }
      }
      
      q1 = q;
    }
    return 0;
    }
}
// BFS TLE
public class Solution {
    public int LadderLength(string beginWord, string endWord, IList<string> wordList) {
        List<String> dict = new List<String>();
    foreach (String word in wordList) dict.Add(word);
    
    if (!dict.Contains(endWord)) return 0;
    
    Queue<String> q = new Queue<String>();
    q.Enqueue(beginWord);
    
    int l = beginWord.Length;
    int steps = 0;
    
    while (q.Count != 0) {
      ++steps;
      for (int s = q.Count; s > 0; --s) {
        String w = q.Dequeue();        
        char[] chs = w.ToCharArray();
        for (int i = 0; i < l; ++i) {
          char ch = chs[i];
          for (char c = 'a'; c <= 'z'; ++c) {
            if (c == ch) continue;
            chs[i] = c;
            String t = new String(chs);         
            if (t.Equals(endWord)) return steps + 1;            
            if (!dict.Contains(t)) continue;            
            dict.Remove(t);            
            q.Enqueue(t);
          }
          chs[i] = ch;
        }
      }
    }
    return 0;
    }
}

// Day 13 Graph Theory
// 997. Find the Town Judge
/*
Intuition:
Consider trust as a graph, all pairs are directed edge.
The point with in-degree - out-degree = N - 1 become the judge.

Explanation:
Count the degree, and check at the end.

Time Complexity:
Time O(T + N), space O(N)
*/
public class Solution {
    public int FindJudge(int n, int[][] trust) {
        int[] degrees = new int[ n+1 ];    
        foreach (int[] t in trust) {
          degrees[t[0]]--;
          degrees[t[1]]++;
        }
        for (int i = 1; i <= n; ++i)
          if (degrees[i] == n - 1) return i;
        return -1;
    }
}

// 1557. Minimum Number of Vertices to Reach All Nodes
/*
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
*/
public class Solution {
    public IList<int> FindSmallestSetOfVertices(int n, IList<IList<int>> edges) {
        List<int> res = new List<int>();
        int[] seen = new int[n];
        foreach (List<int> e in edges)
            seen[e[1]] = 1;
        for (int i = 0; i < n; ++i)
            if (seen[i] == 0)
                res.Add(i);
        return res;
    }
}
// Day 14 Graph Theory
// 1615. Maximal Network Rank
// 
public class Solution {
    public int MaximalNetworkRank(int n, int[][] roads) {
        bool[][] connected = new bool[n][];
        for (int k = 0; k < n; k++) connected[k] = new bool[n];
        int[] cnts = new int[n];
        foreach (int[] r in roads) {
            cnts[r[0]]++;
            cnts[r[1]]++;
            connected[r[0]][r[1]] = true;
            connected[r[1]][r[0]] = true;  // cache if i and j directly connected
        }
        int res = 0;
        for (int i = 0; i < n; i++) 
            for (int j = i + 1; j < n; j++) 
                res = Math.Max(res, cnts[i] + cnts[j] - (connected[i][j] ? 1 : 0));  // loop all pairs
        return res;
    }
}
// 886. Possible Bipartition
/*
Solution: Graph Coloring
Color a node with one color, and color all it’s disliked nodes with another color,
 if can not finish return false.
Time complexity: O(V+E)
Space complexity: O(V+E)
*/
// DFS
public class Solution {
    public bool PossibleBipartition(int n, int[][] dislikes) {
        g_ = new List<int>[n];
        for (int i = 0; i < n; i++) g_[i] = new List<int>();
        foreach(int[] d in dislikes) {
            g_[d[0]-1].Add(d[1]-1);
            g_[d[1]-1].Add(d[0]-1);
        }
        colors_ = new int[n]; 
        Array.Fill(colors_, 0);
        for (int i = 0; i < n; i++)
            if (colors_[i] == 0 && dfs(i, 1) == false)
                return false;
        return true;
    }
private List<int>[] g_;
private int[] colors_;
    bool dfs(int cur, int color) {
        colors_[cur] = color;
        foreach (int nxt in g_[cur]){
            if (colors_[nxt] == color) return false;      
            if (colors_[nxt] == 0 && dfs(nxt, -color) == false) return false;
        }
        return true;
  }
}
// BFS
public class Solution {
    public bool PossibleBipartition(int n, int[][] dislikes) {
        List<int>[] g = new List<int>[n];
        for (int i = 0; i < n; i++) g[i] = new List<int>();
        foreach(int[] d in dislikes) {
            g[d[0]-1].Add(d[1]-1);
            g[d[1]-1].Add(d[0]-1);
        }
        Queue<int> q = new Queue<int>();
        int[] colors = new int[n]; Array.Fill(colors, 0);  // 0: unknown, 1: red, -1: blue
        for (int i = 0; i < n; ++i) {
            if (colors[i] != 0) continue;
            q.Enqueue(i);
            colors[i] = 1;
            while (q.Count != 0) {
                int cur = q.Peek(); q.Dequeue();
                foreach (int nxt in g[cur]) {
                    if (colors[nxt] == colors[cur]) return false;
                    if (colors[nxt] != 0) continue;
                    colors[nxt] = -colors[cur];
                    q.Enqueue(nxt);
                }
            }
        }    
        return true;
    }
}

// 785. Is Graph Bipartite?
// BFS
public class Solution {
    public bool IsBipartite(int[][] graph) {
        // 0(not meet), 1(black), 2(white)
        int[] visited = new int[graph.Length];
        
        for (int i = 0; i < graph.Length; i++) {
            if (graph[i].Length != 0 && visited[i] == 0) {
                visited[i] = 1;
                Queue<int> q = new Queue<int>();
                q.Enqueue(i);
                while(q.Count != 0) {
                    int current = q.Dequeue();
                    foreach (int c in graph[current]) {
                        if (visited[c] == visited[current]) return false;
                        else if (visited[c] == 0) {
                            visited[c] = (visited[current] == 1) ? 2 : 1;
                            q.Enqueue(c);}
                            
                    }
                }                        
            }
        }
        return true;
    }
}
/*
Solution: Graph Coloring
For each node

If has not been colored, color it to RED(1).
Color its neighbors with a different color RED(1) to BLUE(-1) or BLUE(-1) to RED(-1).
If we can finish the coloring then the graph is bipartite. 
All red nodes on the left no connections between them and all blues nodes on the right, 
again no connections between them. red and blue nodes are neighbors.

Time complexity: O(V+E)
Space complexity: O(V)
*/
public class Solution {
    public bool IsBipartite(int[][] graph) {
        int n = graph.Length;
        int[] colors = new int[n];
        for (int i = 0; i < n; ++i)
            if (colors[i] == 0 && coloring(graph, colors, 1, i) != true)
                return false;
        return true;
      }
    private bool coloring(int[][] graph, int[] colors, int color, int node) {    
        if (colors[node] != 0) return colors[node] == color;
        colors[node] = color;
        foreach (int nxt in graph[node])
            if (coloring(graph, colors, -color, nxt) != true) return false;
        return true;
    }
}

// DFS
public class Solution {
    public bool IsBipartite(int[][] graph) {
         int n = graph.Length;int[] colors = new int[n];
        for (int i = 0; i < n; i++) {
            if (colors[i] == 0 && !dfs(graph, colors, i, 1)) 
                return false;            
        }
        return true;
    }
    
    private bool dfs(int[][] graph, int[] colors, int i, int color) {
        colors[i] = color;
        for (int j = 0; j < graph[i].Length; j++) {
            int k = graph[i][j]; // adjacent node
            if (colors[k] == -color) continue;
            if (colors[k] == color || !dfs(graph, colors, k, -color)) return false;
        }
        return true;
    }
}