# Day 1 Binary Search
# 35. Search Insert Position
# nums contains distinct values sorted in ascending order.
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r= 0, len(nums)
        while(l<r):
            m = l+(r-l)/2
            if nums[m]==target :
                return m
            elif nums[m] > target :
                r = m
            else :
                l = m + 1
        
        return l
# 704. Binary Search
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r = 0, len(nums);
        while(l<r):
            m = l+(r-l)/2
            if nums[m] == target:
                return m
            elif nums[m] > target :
                r = m
            else:
                l=m+1
        return -1

# 278. First Bad Version
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution(object):
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        l,r= 0, n
        while(l<r):
            m = l+(r-l)/2
            if isBadVersion(m)==True:
                r = m
            elif isBadVersion(m)==False :
                l = m + 1
        
        return l

# Day 2 Two Pointers
# 189. Rotate Array
# slicing
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if len(nums)>1 and k!=0:
            while k>len(nums):
                k-=len(nums)
            nums[:k],nums[k:] = nums[-k:],nums[:-k] 

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        if k % len(nums) and len(nums) > 1:
            k = k % len(nums)
            nums[:] = nums[-k:] + nums[:-k]

# The basic idea is that, for example, nums = [1,2,3,4,5,6,7] and k = 3, 
# first we reverse [1,2,3,4], it becomes[4,3,2,1];
# then we reverse[5,6,7], it becomes[7,6,5],
# finally we reverse the array as a whole, it becomes[4,3,2,1,7,6,5] ---> [5,6,7,1,2,3,4].

# Reverse is done by using two pointers, one point at the head and the other point at the tail,
# after switch these two, these two pointers move one position towards the middle.

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        def twopt(arr, i, j):
            while (i < j):
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
            return arr
        
        if k > len(nums):
            k %= len(nums)
            
        if (k > 0):
            twopt(nums, 0, len(nums) - 1)  # rotate entire array
            twopt(nums, 0, k - 1)          # rotate array upto k elements
            twopt(nums, k, len(nums) - 1)  # rotate array from k to end of array

# 977. Squares of a Sorted Array
class Solution(object):
    def sortedSquares(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        answer = [0] * len(nums)
        l, r = 0, len(nums) - 1
        while l <= r:
            if abs(nums[l]) > abs(nums[r]):
                answer[r - l] = nums[l] * nums[l]
                l += 1
            else:
                answer[r - l] = nums[r] * nums[r]
                r -= 1
        return answer

# Day 3 Two Pointers
# 283. Move Zeroes
# Time complexity: O(n)
# Space complexity: O(1).
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero = 0  # records the position of "0"
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[zero], nums[i] = nums[i], nums[zero]
                zero += 1

class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        l = 0
        r = len(nums)
        while l < r:
            if nums[l] == 0:
                del nums[l]
                nums.append(0)
                r -= 1
            else:
                l += 1

# 167. Two Sum II - Input Array Is Sorted 
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        l,r = 0, len(numbers)-1
        while(l<r):
            if numbers[l]+numbers[r]==target:
                return [l+1,r+1]
            elif numbers[l]+numbers[r]>target:
                r-=1
            else:
                l+=1
        return []

# Day 4 Two Pointers
# 344. Reverse String
class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        i  = 0
        j  = len(s) -1
        while i < j:
            s[i],s[j] = s[j],s[i]
            i += 1
            j -= 1

# 557. Reverse Words in a String III
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
    # splict the string
    # reverse the string list
    # join together the string list as string
    # reverse the whole string
        return join(s.split()[::-1])[::-1]

# Day 5 Two Pointers
# 876. Middle of the Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # While slow moves one step forward, fast moves two steps forward.
        # Finally, when fast reaches the end, slow happens to be in the middle of the linked list.
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

# 19. Remove Nth Node From End of List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        fast, slow = head, head
        # don't care about the variable _ we just want to do the loop.
        for _ in range(n):
            fast = fast.next
        # if fast is None, that means the node that you need to remove is the head. Hence, you return head.next as the head of the list.
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head

# Day 6 Sliding Window
# 3. Longest Substring Without Repeating Characters
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        used = {}
        max_length = 0 
        start = -1
        for i, c in enumerate(s):
            if c in used and used[c]>start:
                start = used[c] 
            max_length = max(max_length, i-start)
            used[c] = i

# 567. Permutation in String
# The only thing we care about any particular substring in s2 
# is having the same number of characters as in the s1.
# So we create a hashmap with the count of every character in the string s1.
# Then we slide a window over the string s2 and decrease the counter for characters
# that occurred in the window. As soon as all counters in the hashmap get to zero
# that means we encountered the permutation.
# Time: O(n) - linear for window sliding and counter
# Space: O(1) - conctant for dictionary with the maximum 26 pairs (English alphabet)
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        cntr, w = Counter(s1), len(s1)   

        for i in range(len(s2)):
            if s2[i] in cntr: 
                cntr[s2[i]] -= 1
            if i >= w and s2[i-w] in cntr: 
                cntr[s2[i-w]] += 1

            if all([cntr[i] == 0 for i in cntr]): # see optimized code below
                return True

        return False

# Optimized:
# We can use an auxiliary variable to count a number of characters whose 
# frequency gets to zero during window sliding. 
# That helps us to avoid iterating over the hashmap for every cycle tick 
# to check whether frequencies turned into zero.
class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        cntr, w, matched = Counter(s1), len(s1), 0   

        for i in range(len(s2)):
            if s2[i] in cntr: 
                cntr[s2[i]] -= 1
                if cntr[s2[i]] == 0:
                    matched += 1
            if i >= w and s2[i-w] in cntr: 
                if cntr[s2[i-w]] == 0:
                    matched -= 1
                cntr[s2[i-w]] += 1

            if matched == len(cntr):
                return True

        return False

# Day 7 Breadth-First Search / Depth-First Search

# 733. Flood Fill
# Time complexity: O(m*n), space complexity: O(1). m is number of rows, n is number of columns.
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        def fill(image, sr, sc, color, newColor):
            if sr < 0 or sr >= len(image) or sc < 0 or sc >= len(image[0]) or image[sr][sc] != color : return 
            image[sr][sc] = newColor
            fill(image, sr + 1, sc, color, newColor)
            fill(image, sr - 1, sc, color, newColor)
            fill(image, sr, sc + 1, color, newColor)
            fill(image, sr, sc - 1, color, newColor)
        
        if image[sr][sc] != newColor: 
            fill(image, sr, sc, image[sr][sc], newColor)
        return image
# Approach #1: Depth-First Search
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        R, C = len(image), len(image[0])
        color = image[sr][sc]
        if color == newColor: return image
        def dfs(r, c):
            if image[r][c] == color:
                image[r][c] = newColor
                if r >= 1: dfs(r-1, c)
                if r+1 < R: dfs(r+1, c)
                if c >= 1: dfs(r, c-1)
                if c+1 < C: dfs(r, c+1)

        dfs(sr, sc)
        return image

# 695. Max Area of Island
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def dfs(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j]:
                grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0
        
        maximum = 0
        for i in range (len(grid)):
            for j in range(len(grid[0])):
                 if grid[i][j]:
                    maximum = max(dfs(i,j),maximum)
        return maximum

class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def dfs(i, j):
            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j]:
                grid[i][j] = 0
                return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
            return 0

        areas = [dfs(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j]]
        
        return max(areas) if areas else 0
# Approach #1: Depth-First Search (Recursive)
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        seen = set()
        def area(r, c):
            if not (0 <= r < len(grid) and 0 <= c < len(grid[0])
                    and (r, c) not in seen and grid[r][c]):
                return 0
            seen.add((r, c))
            return (1 + area(r+1, c) + area(r-1, c) +
                    area(r, c-1) + area(r, c+1))

        return max(area(r, c)
                   for r in range(len(grid))
                   for c in range(len(grid[0])))

# Approach #2: Depth-First Search (Iterative)
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        seen = set()
        ans = 0
        for r0, row in enumerate(grid):
            for c0, val in enumerate(row):
                if val and (r0, c0) not in seen:
                    shape = 0
                    stack = [(r0, c0)]
                    seen.add((r0, c0))
                    while stack:
                        r, c = stack.pop()
                        shape += 1
                        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                                    and grid[nr][nc] and (nr, nc) not in seen):
                                stack.append((nr, nc))
                                seen.add((nr, nc))
                    ans = max(ans, shape)
        return ans

# Day 8 Breadth-First Search / Depth-First Search

# 617. Merge Two Binary Trees

# Tree Traversal

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        if not root1 and not root2: return None
        elif not root1: return root2
        elif not root2: return root1
        ans = TreeNode(root1.val + root2.val)
        ans.left = self.mergeTrees(root1.left, root2.left)
        ans.right = self.mergeTrees(root1.right, root2.right)
        return ans

class Solution(object):
    def mergeTrees(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: TreeNode
        """
        if not root1 and not root2: return None
        ans = TreeNode((root1.val if root1 else 0) + (root2.val if root2 else 0))
        ans.left = self.mergeTrees(root1.left if root1 else None, root2.left if root2 else None)
        ans.right = self.mergeTrees(root1.right if root1 else None, root2.right if root2 else None)
        return ans

# 116. Populating Next Right Pointers in Each Node
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        curr = root
        while curr and curr.left:
            left = curr.left
            while curr:
                curr.left.next = curr.right
                curr.right.next = curr.next.left if curr.next else None
                curr = curr.next
            curr = left
        return root

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root: return None
        if root.left != None:
            root.left.next = root.right
            if root.next != None:
                root.right.next = root.next.left  
                
        self.connect(root.left)
        self.connect(root.right)
        return root


class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root and root.left and root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            self.connect(root.left)
            self.connect(root.right)

        return root

# Day 9 Breadth-First Search / Depth-First Search

# 542. 01 Matrix
# DP solution
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
        

# 994. Rotting Oranges
# from collections import deque
# Time complexity: O(rows * cols) -> each cell is visited at least once
# Space complexity: O(rows * cols) -> in the worst case if all the oranges are rotten they will be added to the queue
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        # number of rows
        rows = len(grid)
        if rows == 0:  # check if grid is empty
            return -1
        
        # number of columns
        cols = len(grid[0])
        
        # keep track of fresh oranges
        fresh_cnt = 0
        
        # queue with rotten oranges (for BFS)
        rotten = deque()
        
        # visit each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    # add the rotten orange coordinates to the queue
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    # update fresh oranges count
                    fresh_cnt += 1
        
        # keep track of minutes passed.
        minutes_passed = 0
        
        # If there are rotten oranges in the queue and there are still fresh oranges in the grid keep looping
        while rotten and fresh_cnt > 0:

            # update the number of minutes passed
            # it is safe to update the minutes by 1, since we visit oranges level by level in BFS traversal.
            minutes_passed += 1
            
            # process rotten oranges on the current level
            for _ in range(len(rotten)):
                x, y = rotten.popleft()
                
                # visit all the adjacent cells
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    # calculate the coordinates of the adjacent cell
                    xx, yy = x + dx, y + dy
                    # ignore the cell if it is out of the grid boundary
                    if xx < 0 or xx == rows or yy < 0 or yy == cols:
                        continue
                    # ignore the cell if it is empty '0' or visited before '2'
                    if grid[xx][yy] == 0 or grid[xx][yy] == 2:
                        continue
                        
                    # update the fresh oranges count
                    fresh_cnt -= 1
                    
                    # mark the current fresh orange as rotten
                    grid[xx][yy] = 2
                    
                    # add the current rotten to the queue
                    rotten.append((xx, yy))

        
        # return the number of minutes taken to make all the fresh oranges to be rotten
        # return -1 if there are fresh oranges left in the grid (there were no adjacent rotten oranges to make them rotten)
        return minutes_passed if fresh_cnt == 0 else -1


class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        visit, curr = set(), deque()
		# find all fresh and rotten oranges
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    visit.add((i, j))
                elif grid[i][j] == 2:
                    curr.append((i, j))
        result = 0
        while visit and curr:
			# BFS iteration
            for _ in range(len(curr)):
                i, j = curr.popleft()  # obtain recent rotten orange
                for coord in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if coord in visit:  # check if adjacent orange is fresh
                        visit.remove(coord)
                        curr.append(coord)
            result += 1
		# check if fresh oranges remain and return accordingly
        return -1 if visit else result


class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n, queue, fresh = len(grid), len(grid[0]), deque(), 0
        for i,j in product(range(m), range(n)):
            if grid[i][j] == 2: queue.append((i,j))
            if grid[i][j] == 1: fresh += 1
        dirs = [[1,0],[-1,0],[0,1],[0,-1]]
        levels = 0
        
        while queue:
            levels += 1
            for _ in range(len(queue)):
                x, y = queue.popleft()
                for dx, dy in dirs:
                    if 0<=x+dx<m and 0<=y+dy<n and grid[x+dx][y+dy] == 1:
                        fresh -= 1
                        grid[x+dx][y+dy] = 2
                        queue.append((x+dx, y+dy))
                        
        return -1 if fresh != 0 else max(levels-1, 0)

# Day 11 Recursion / Backtracking

# 77. Combinations
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        ans = []
        cur = []
        def dfs(s):
            if len(cur)==k:
                ans.append(cur[:])
                return
            for i in range(s,n):
                cur.append(i+1)
                dfs(i+1)
                cur.pop()
        
        dfs(0)
        return ans

class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """

        output = []
    
        def backtracking(start, pairs):
            if len(pairs) == k:
                output.append(pairs[:])
                return

            for i in range(start, n+1):
                backtracking(i+1, pairs + [i])

        backtracking(1, [])
        return output

# 46. Permutations
# Solution: DFS
# Time complexity: O(n!)
# Space complexity: O(n)
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        ans = []
        used = n*[None]
        path =[]
        def dfs(d):
            if d== n:
                ans.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]: 
                    continue
                used[i] = 1
                path.append(nums[i])
                dfs(d+1)                
                used[i]=0
                path.pop()
                i+=1
        dfs(0)
        return ans

# 784. Letter Case Permutation
class Solution(object):
    def letterCasePermutation(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        ans = []
 
        def dfs(S, i, n):
          if i == n:
            ans.append(''.join(S))
            return

          dfs(S, i + 1, n)      
          if not S[i].isalpha(): return      
          S[i] = chr(ord(S[i]) ^ (1<<5)) # ord('A')    # 65
          dfs(S, i + 1, n)
          S[i] = chr(ord(S[i]) ^ (1<<5)) # chr(97)     # a

        dfs(list(s), 0, len(s))
        return ans

# Day 12 Dynamic Programming

# 70. Climbing Stairs
class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        one, two,curr=1,1,1
        for i in range(2,n+1):
            curr = one+two
            two = one
            one = curr
        return curr

# 198. House Robber
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: 
            return 0
        dp2 = 0
        dp1 = 0
        for i in range(0,len(nums)):
            dp = max(dp2+nums[i],dp1)
            dp2 = dp1
            dp1 = dp
            
        return dp1

# 120. Triangle

class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        n = len(triangle)
        for i in range(n):
            for j in range(i+1):
                if i==0 and j==0: continue
                elif j==0: triangle[i][j] += triangle[i-1][j]
                elif j==i: triangle[i][j] += triangle[i-1][j-1]
                else: triangle[i][j] += min(triangle[i-1][j],triangle[i-1][j-1])
        return sorted(triangle[-1])[0]

# Day 13 Bit Manipulation
# 231. Power of Two
'''
Solution - (Bit-Trick)

There's a nice bit-trick that can be used to check if a number is power of 2 efficiently. As already seen above, n will only have 1 set bit if it is power of 2. Then, we can AND (&) n and n-1 and if the result is 0, it is power of 2. This works because if n is power of 2 with ith bit set, then in n-1, i will become unset and all bits to right of i will become set. Thus the result of AND will be 0.

If n is a power of 2:
n    = 8 (1000)
n-1  = 7 (0111)
----------------
&    = 0 (0000)         (no set bit will be common between n and n-1)

If n is not a power of 2:
n    = 10 (1010)
n-1  =  9 (1001)
-----------------
&    =  8 (1000)         (atleast 1 set bit will be common between n and n-1)

Time Complexity : O(1)
Space Complexity : O(1)
'''
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n > 0 and n & (n-1) == 0
'''
Solution - (Math)
Only a power of 2 will be able to divide a larger power of 2. Thus, we can take the largest power of 2 for our given range and check if n divides it
Time Complexity : O(1)
Space Complexity : O(1)
'''
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        return n > 0 and (1 << 31) % n == 0
'''
Solution - (Recursive)
If a number is power of two, it can be recursively divided by 2 till it becomes 1
If the start number is 0 or if any intermediate number is not divisible by 2, we return false
Time Complexity : O(logn), where n is the given input number
Space Complexity : O(logn), required for recursive stack
'''
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0: return False
        return n == 1 or (n % 2 == 0 and self.isPowerOfTwo(n // 2))
'''
Solution - (Iterative)
The same solution as above but done iteratively
Time Complexity : O(logn), where n is the given input number
Space Complexity : O(1)
'''
class Solution(object):
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n == 0: return False
        while n % 2 == 0:
            n /= 2
        return n == 1

# 191. Number of 1 Bits
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        c = 0
        while n:
            n &= n - 1
            c += 1
        return c

# Day 14 Bit Manipulation
# 190. Reverse Bits
# One small thing is the plus operator can be replaced by "bitwise or", aka "|". 
# However i found plus is more readable and fast in python.
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        if n == 0 : return 0
        ans = 0
        for i in range(32):
            ans = (ans << 1) + (n & 1)
            n >>= 1
        return ans

# 136. Single Number

class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1,len(nums)):
            nums[0] ^= nums[i]
        return nums[0]
        
# Algorithm II
# Day 1 Binary Search
# 34. Find First and Last Position of Element in Sorted Array
 
# 33. Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)
        while l < r:
            m = l+(r-l) # 2
            if nums[m] == target:
                return m
            if nums[0] <= target < nums[m] or target < nums[m] < nums[0] or nums[m] < nums[0] <= target:
                r = m
            else:
                l = m+1

        return -1

# 74. Search a 2D Matrix

# Day 2 Binary Search
# 153. Find Minimum in Rotated Sorted Array

class Solution(object):
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = 0
        r = len(nums) - 1
        while l < r:
            m = l + (r - l) # 2
            if nums[m] < nums[r]:
                # the mininum is in the left part
                r = m
            elif nums[m] > nums[r]:
                # the mininum is in the right part
                l = m + 1
        return nums[l]

# 162. Find Peak Element
'''
Solution 2: Binary Search

Binary search starting with left = 0, right = n-1, mid = (left + right) / 2
If nums[mid-1] < nums[mid] && nums[mid] > nums[mid+1] return mid as peak.
Else if nums[mid-1] < nums[mid] then search peak on the right side.
Else search pick on the left side.

Complexity

Time: O(logN)
Space: O(1)

'''
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l, r = 0, len(nums) - 1
        while l <= r:
            m = l +(r-l) # 2
            if (m == 0 or nums[m-1] < nums[m]) and (m == len(nums)-1 or nums[m] > nums[m+1]):  # Found peak
                return m
            if m == 0 or nums[m-1] < nums[m]:  # Find peak on the right
                l = m + 1
            else:  # Find peak on the left
                r = m - 1
        return -1

class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1: return 0
        l, r = 0, len(nums) - 1
        while l < r:
            m = l +(r-l) # 2
            if nums[m] > nums[m+1]:  # Found peak
                r = m
            else:  # Find peak on the right
                l = m + 1
        
        return l

# Day 3 Two Pointers
# 82. Remove Duplicates from Sorted List II

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # sentinel
        sentinel = ListNode(0, head)

        # predecessor = the last node 
        # before the sublist of duplicates
        pred = sentinel
        
        while head:
            # if it's a beginning of duplicates sublist 
            # skip all duplicates
            if head.next and head.val == head.next.val:
                # move till the end of duplicates sublist
                while head.next and head.val == head.next.val:
                    head = head.next
                # skip all duplicates
                pred.next = head.next 
            # otherwise, move predecessor
            else:
                pred = pred.next 
                
            # move forward
            head = head.next
            
        return sentinel.next

# 15. 3Sum
# Solution 2: Sorting + Two pointers
# Time complexity: O(nlogn + n^2)
# Space complexity: O(1)
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        for i in range(len(nums)-2):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l, r = i+1, len(nums)-1
            while l < r:
                if nums[i] + nums[l] + nums[r] < 0:
                    l +=1 
                elif nums[i] + nums[l] + nums[r] > 0:
                    r -= 1
                else:
                    res.append((nums[i], nums[l], nums[r]))
                    l += 1, r -= 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                    while l < r and nums[r] == nums[r+1]:
                        r -= 1
                    
        return res

# Day 4 Two Pointers
# 844. Backspace String Compare
'''
Approach #2: Two Pointer
Time Complexity: O(M + N)O(M+N), where M, NM,N are the lengths of S and T respectively.
Space Complexity: O(1)O(1).
'''
class Solution(object):
    def backspaceCompare(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        def F(S):
            skip = 0
            for x in reversed(S):
                if x == '#':
                    skip += 1
                elif skip:
                    skip -= 1
                else:
                    yield x

        return all(x == y for x, y in itertools.izip_longest(F(s), F(t)))

class Solution(object):
    def backspaceCompare(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        i, j = len(s) - 1, len(t) - 1
        backS = backT = 0
        while i >= 0  or j >= 0 :
            while i >= 0:
                if s[i] == '#':
                    backS += 1 
                    i -= 1
                elif backS: 
                    backS-=1
                    i -= 1
                else:
                    break
            while j >= 0:
                if t[j] == '#':
                    backT += 1 
                    j -= 1
                elif backT:
                    backT-=1
                    j -= 1
                else:
                    break

            if i >= 0 and j >= 0 and s[i] != t[j]:
                return False
            if (i >= 0) != (j >= 0):
                return False
            i, j = i - 1, j - 1
        return True


# 986. Interval List Intersections
'''
Solution: Two pointers
Time complexity: O(m + n)
Space complexity: O(1)
'''
class Solution(object):
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        i, j, ans = 0, 0, []
        while i < len(firstList) and j < len(secondList):
            s = max(firstList[i][0],  secondList[j][0])
            e = min(firstList[i][1],  secondList[j][1])
            if s <= e: ans.append([s, e])
            if firstList[i][1] <  secondList[j][1]:
                i += 1
            else:
                j += 1
        return ans

# 11. Container With Most Water
'''
Two pointers
Time complexity: O(n)
Space complexity: O(1)
'''
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        l, r, ans = 0, len(height) - 1, 0
        while l < r:
            h = min(height[l], height[r])
            ans = max(ans, h * (r - l))
            if height[l] < height[r]:l+=1
            else:r-=1
        
        return ans

# Day 5 Sliding Window
# 438. Find All Anagrams in a String
'''
This problem is an advanced version of 567. Permutation in String.
Firstly, we count the number of characters needed in p string.
Then we sliding window in the s string:
Let l control the left index of the window, r control the right index of the window (inclusive).
Iterate r in range [0..n-1].
When we meet a character c = s[r], we decrease the cnt[c] by one by cnt[c]--.
If the cnt[c] < 0, it means our window contains char c with the number more than in p, which is invalid.
So we need to slide left to make sure cnt[c] >= 0.
If r - l + 1 == p.length then we already found a window which is perfect match with string p. 
WHY? Because window has length == p.length and window doesn't contains any characters 
which is over than the number in p.

Complexity
Time: O(|s| + |p|)
Space: O(1)
'''
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        cnt = Counter(p)
        
        ans = []
        l = 0
        for r, c in enumerate(s):
            cnt[c] -= 1
            while cnt[c] < 0:  # If number of characters `c` is more than our expectation
                cnt[s[l]] += 1  # Slide left until cnt[c] == 0
                l += 1
            if r - l + 1 == len(p):  #  If we already filled enough `p.length()` chars
                ans.append(l)  # Add left index `l` to our result
                
        return ans

# 713. Subarray Product Less Than K
# Approach #2: Sliding Window 
'''
The idea is that we keep 2 points l (initial value = 0) point to the left most of window, 
r point to current index of nums.
We use product (initial value = 1) to keep the product of numbers in the window range.
While iterating r in [0...n-1], we calculate number of subarray 
which ends at nums[r] has product less than k.
product *= nums[r].
While product > k && l <= r then we slide l by one
Now product < k, then there is r-l+1 subarray which ends at nums[r] has product less than k.
'''
# Complexity:
# Time: O(N), where N <= 3*10^4 is length of nums array.
# Space: O(1)
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        if k <= 1: return 0
        prod = 1
        ans = left = 0
        for right, val in enumerate(nums):
            prod *= val
            while prod >= k:
                prod /= nums[left]
                left += 1
            ans += right - left + 1
        return ans

# 209. Minimum Size Subarray Sum
'''
Sliding Window
Intuition
Shortest Subarray with Sum at Least K
Actually I did this first, the same prolem but have negatives.
I suggest solving this prolem first then take 862 as a follow-up.

Explanation
The result is initialized as res = n + 1.
One pass, remove the value from sum s by doing s -= A[j].
If s <= 0, it means the total sum of A[i] + ... + A[j] >= sum that we want.
Then we update the res = min(res, j - i + 1)
Finally we return the result res

Complexity
Time O(N)
Space O(1)
'''
class Solution(object):
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        i, res = 0, len(nums) + 1
        for j in range(len(nums)):
            target -= nums[j]
            while target <= 0:
                res = min(res, j - i + 1)
                target += nums[i]
                i += 1
        return res % (len(nums) + 1)

# Day 6 Breadth-First Search / Depth-First Search
# 200. Number of Islands
'''
Idea: DFS
Use DFS to find a connected component (an island) and mark all the nodes to 0.

Time complexity: O(mn)
Space complexity: O(mn)
'''
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        m = len(grid)
        if m == 0: return 0
        n = len(grid[0])
        
        ans = 0
        for y in range(m):
            for x in range(n):
                if grid[y][x] == '1':
                    ans += 1
                    self.__dfs(grid, x, y, n, m)
        return ans
    
    def __dfs(self, grid, x, y, n, m):
        if x < 0 or y < 0 or x >=n or y >= m or grid[y][x] == '0':
            return
        grid[y][x] = '0'
        self.__dfs(grid, x + 1, y, n, m)
        self.__dfs(grid, x - 1, y, n, m)
        self.__dfs(grid, x, y + 1, n, m)
        self.__dfs(grid, x, y - 1, n, m)
# 547. Number of Provinces
# Solution: DFS
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

# Day 7 Breadth-First Search / Depth-First Search

# 117. Populating Next Right Pointers in Each Node II
# Solution : BFS
# Complexity
# Time: O(N), where N <= 6000 is the number of nodes in the binary tree.
# Space: O(N)

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if root == None: return root
        q = deque([root])
        while q:
            prev = None
            for _ in range(len(q)):
                cur = q.popleft()
                if prev != None:
                    prev.next = cur
                prev = cur
                if cur.left != None:
                    q.append(cur.left)
                if cur.right != None:
                    q.append(cur.right)
        return root

# O(1) Space Approach
# In addition to this, there is a follow-up question asking to solve this problem using constant extra space. 
# There is an additional hint to maybe use recursion for this and the extra call stack is assumed to be O(1) space

# The code will track the head at each level and use that not null head to define the next iteration.
# Time = O(N) - iterate through all the nodes
# Space = O(1) - No additional space

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""
class Solution(object):
    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return None
        
        curr=root
        dummy=Node(0)        
        head=root        
        
        while head:
            curr=head # initialize current level's head
            prev=dummy # init prev for next level linked list traversal
			# iterate through the linked-list of the current level and connect all the siblings in the next level
            while curr:  
                if curr.left:
                    prev.next=curr.left
                    prev=prev.next
                if curr.right:
                    prev.next=curr.right
                    prev=prev.next                                                
                curr=curr.next
            head=dummy.next # update head to the linked list of next level
            dummy.next=None # reset dummy node
        return root
        
# 572. Subtree of Another Tree

# Complexity
# Time: O(M * N), where M is the number of nodes in binary tree root, 
# N is the number of nodes in binary tree subRoot
# Space: O(H), where H is the height of binary tree root

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    
    def isEqual(self, root1, root2):
        if root1 == None and root2 == None: return True
        if root1 == None or root2 == None: return False
        if root1.val != root2.val : return False
        return self.isEqual(root1.left, root2.left) and self.isEqual(root1.right, root2.right)
    
    def isSubtree(self, root, subRoot):
        """
        :type root: TreeNode
        :type subRoot: TreeNode
        :rtype: bool
        """    
        if root == None: return False
        if self.isEqual(root, subRoot): return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

# Day 8 Breadth-First Search / Depth-First Search

# 1091. Shortest Path in Binary Matrix
# BFS
'''
Complexity
Time: O(M * N), where M <= 100 is the number of rows, N <= 100 is number of columns in the matrix.
BFS cost O(E + V), where E = 8 * V is number of edges, V = M*N is number of vertices.
So total complexity: O(8V + V) = O(9V) = O(9 * M*N) ~ O(M*N)
Space: O(M * N)
'''
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

# DFS
""" O(N^2)TS """
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        def fn(level, deep):
            (max_, max_) in level and res.append(deep + 1)
            #print(level)
            level and fn([kid for y, x in level for kid in ((y, x + 1), (y, x - 1), (y + 1, x), (y + 1, x + 1), (y + 1, x - 1), (y - 1, x), (y - 1, x + 1), (y - 1, x - 1)) if grid.pop(kid, False)], deep + 1)

        grid, max_, res = {(y, x): 1 - val for y, row in enumerate(grid) for x, val in enumerate(row)}, len(grid) - 1, [-1]
        return fn([(0, 0)], 0) or res.pop() if grid.pop((0, 0)) else -1
'''
Unfortunately, using dfs you'd have to try every possible path to the end.
You'd have to mark a cell as unvisited after recurring the neighbors (after the for-loop). 
But doing this in this problem would lead to TLE.
For this reason, BFS is the best choice here.
'''
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        m, n = len(grid), len(grid[0])
        
        dist = [[float('inf')] * n for _ in range(m)]
        dist[0][0] = 1
        
        if grid[0][0]==1 or grid[m-1][n-1]==1: 
            return -1           
        self.dfs(grid, dist, 0, 0)               
        return dist[m-1][n-1] if dist[m-1][n-1] != float('inf') else -1
    def dfs(self,grid, dist, r, c):
            m, n = len(grid), len(grid[0])
            d0 = dist[r][c]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    x , y = r + i , c + j
                    if x >= 0 and x < m and y >= 0 and y < n : 
                        if grid[r][c] == 1: continue
                        d1 = dist[x][y]
                        if d1 < d0:
                            dist[r][c] = d1 + 1
                            self.dfs(grid, dist, r, c) # <= RuntimeError: maximum recursion depth exceeded
                            return
                        elif d1 > d0 + 1:
                            dist[x][y] = d0 + 1
                            self.dfs(grid, dist, x, y)
# 130. Surrounded Regions
# DFS 
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return 
        for i in [0, len(board)-1]:
            for j in range(len(board[0])):
                self.dfs(board, i, j)   
        for i in range(len(board)):
            for j in [0, len(board[0])-1]:
                self.dfs(board, i, j)
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                elif board[i][j] == '.':
                    board[i][j] = 'O'
                
    def dfs(self, board, i, j):
        if 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] == 'O':
            board[i][j] = '.'
            self.dfs(board, i+1, j)
            self.dfs(board, i-1, j)
            self.dfs(board, i, j+1)
            self.dfs(board, i, j-1)

# BFS
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        queue = collections.deque([])
        for r in range(len(board)):
            for c in range(len(board[0])):
                if (r in [0, len(board)-1] or c in [0, len(board[0])-1]) and board[r][c] == "O":
                    queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            if 0<=r<len(board) and 0<=c<len(board[0]) and board[r][c] == "O":
                board[r][c] = "."
                queue.extend([(r-1, c),(r+1, c),(r, c-1),(r, c+1)])
        
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == ".":
                    board[r][c] = "O"

if not board or not board[0]:
            return
        R, C = len(board), len(board[0])
        if R<=2 or C<=2:
            return
        queue = collections.deque()
        for r in range(R):
            queue.append((r, 0))
            queue.append((r, C - 1))

        for c in range(len(board[0])):
            queue.append((0, c))
            queue.append((R - 1, c))

        while queue:
            r, c = queue.popleft()
            # print(r, c)
            if 0 <= r < R and 0 <= c < C and board[r][c] == 'O':
                board[r][c] = 'N'
                queue.append((r-1, c))
                queue.append((r + 1, c))
                queue.append((r, c + 1))
                queue.append((r, c - 1))

        for r in range(R):
            for c in range(C):
                if board[r][c] == 'N':
                    board[r][c] = 'O'
                else:
                    board[r][c] = 'X'

# 797. All Paths From Source to Target
''' Backtracking
If it asks just the number of paths, generally we can solve it in two ways.
    Count from start to target in topological order.
    Count by dfs with memo.
    Both of them have time O(Edges) and O(Nodes) space. Let me know if you agree here.
I didn't do that in this problem, for the reason that it asks all paths. I don't expect memo to save much time. (I didn't test).
Imagine the worst case that we have node-1 to node-N, and node-i linked to node-j if i < j.
There are 2^(N-2) paths and (N+2)*2^(N-3) nodes in all paths. We can roughly say O(2^N).
'''
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

# Day 9 Recursion / Backtracking
# 78. Subsets

# Binary / Bit operation
# Time complexity: O(n * 2^n)
# Space complexity: O(1)
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        n = len(nums)
        for i in range(1 << n):
            tmp = []
            for j in range(n):
                if i & 1 << j:  # if i >> j & 1:
                    tmp.append(nums[j])
            res.append(tmp)
        return res

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        return [[nums[i] for i in range(n) if s & 1 << i > 0] for s in range(1 << n)]

# DFS + Backtracking
# Time complexity: O(n * 2^n)
# Space complexity: O(n)
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ans = []
        def dfs(n, s, cur):
            if n == len(cur):
                ans.append(cur[:])
                return
            for i in range(s, len(nums)):
                cur.append(nums[i])
                dfs(n, i + 1, cur)
                cur.pop()
        for i in range(len(nums) + 1):
            dfs(i, 0, [])
        return ans

# 90. Subsets II
'''
Solution: DFS
The key to this problem is how to remove/avoid duplicates efficiently.
For the same depth, among the same numbers, only the first number can be used.

Time complexity: O(2^n * n)
Space complexity: O(n)
'''
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(index, curSubset):
            ans.append(curSubset[::])

            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i - 1]: continue  # Skip duplicates
                curSubset.append(nums[i])
                backtrack(i + 1, curSubset)
                curSubset.pop()

        nums.sort()
        ans = []
        backtrack(0, [])
        return ans

class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ret = []
        self.dfs(sorted(nums), [], ret)
        return ret
    
    def dfs(self, nums, path, ret):
        ret.append(path)
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.dfs(nums[i+1:], path+[nums[i]], ret)

# Day 10 Recursion / Backtracking

# 47. Permutations II
# backtracking solution
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        nums.sort()
        self.dfs(nums, [], res)
        return res

    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)

# Approach : Backtracking with Groups of Numbers
# Time Complexity: k-permutations_of_N or partial permutation.
# Space Complexity: O(N)
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        results = []
        def backtrack(comb, counter):
            if len(comb) == len(nums):
                # make a deep copy of the resulting permutation,
                # since the permutation would be backtracked later.
                results.append(list(comb))
                return

            for num in counter:
                if counter[num] > 0:
                    # add this number into the current combination
                    comb.append(num)
                    counter[num] -= 1
                    # continue the exploration
                    backtrack(comb, counter)
                    # revert the choice for the next exploration
                    comb.pop()
                    counter[num] += 1

        backtrack([], Counter(nums))
        
        return results

# 39. Combination Sum
# DFS
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(candidates, target, s, curr, ans):
            if target == 0: 
                ans.append(curr[:])
                return
            
            for i in range(s, len(candidates)):
                if candidates[i] > target: return
                curr.append(candidates[i])
                dfs(candidates, target - candidates[i], i, curr, ans)
                curr.pop()
        
        ans = []        
        candidates.sort()        
        dfs(candidates, target, 0, [], ans)
        
        return ans

# 40. Combination Sum II
# DFS
# Time complexity: O(2^n)
# Space complexity: O(kn)
# How to remove duplicates?
# 1. Use set
# 2. Disallow same number in same depth 
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(candidates, target, s, curr, ans):
            if target == 0: 
                ans.append(curr[:])
                return
            
            for i in xrange(s, len(candidates)):
                if candidates[i] > target: return
                if i>s and candidates[i] == candidates[i-1]: continue
                curr.append(candidates[i])
                dfs(candidates, target - candidates[i], i+1, curr, ans)
                curr.pop()
        
        ans = []        
        candidates.sort()        
        dfs(candidates, target, 0, [], ans)
        
        return ans
# Use set <- TLE
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(candidates, target, s, curr, ans):
            if target == 0: 
                ans.add(tuple(curr[:])) # curr[:] could cost error
                return
            
            for i in xrange(s, len(candidates)):
                if candidates[i] > target: return
                curr.append(candidates[i])
                dfs(candidates, target - candidates[i], i+1, curr, ans)
                curr.pop()
        
        ans = set()        
        candidates.sort()        
        dfs(candidates, target, 0, [], ans)
        
        return list(ans)
# Day 11 Recursion / Backtracking

# 17. Letter Combinations of a Phone Number

# 22. Generate Parentheses
# DFS
# Solution: DFS
# Time complexity: O(2^n)
# Space complexity: O(k + n)
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        if n > 0: self.dfs(res, n, n, '')

        return res
        
    def dfs(self, res, left, right, path):
        if left == 0 and right == 0:
            res.append(path)
            return
        if right < left:  return
        if left > 0:
            self.dfs(res, left - 1, right, path + '(')
        if right > 0:
            self.dfs(res, left, right - 1, path + ')')

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        res = []
        self.dfs(res, n, n, '')

        return res
        
    def dfs(self, res, left, right, path):
        if left == 0 and right == 0:
            res.append(path)
            return
        if left > 0:
            self.dfs(res, left - 1, right, path + '(')
        if left < right:
            self.dfs(res, left, right - 1, path + ')')

# 79. Word Search
# DFS
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        if not board: return False
        h, w = len(board), len(board[0])

        def search(d, x, y):
            if x < 0 or x == w or y < 0 or y == h or word[d] != board[y][x]: return False
            if d == len(word) - 1: return True

            cur = board[y][x]
            board[y][x] = ''
            found = search(d + 1, x + 1, y) or search(d + 1, x - 1, y) or search(d + 1, x, y + 1) or search(d + 1, x, y - 1)
            board[y][x] = cur
            return found

        return any(search(0, j, i) for i in range(h) for j in range(w)) 

# Day 12 Dynamic Programming
# 213. House Robber II
'''
This problem can be seen as follow-up question for problem 198. House Robber. 
Imagine, that we can already solve this problem: for more detailes please see my post:
https:#leetcode.com/problems/house-robber/discuss/846004/Python-4-lines-easy-dp-solution-explained

Now, what we have here is circular pattern. 
Imagine, that we have 10 houses: a0, a1, a2, a3, ... a9: Then we have two possible options:

Rob house a0, then we can not rob a0 or a9 and we have a2, a3, ..., a8 range to rob
Do not rob house a0, then we have a1, a2, ... a9 range to rob.
Then we just choose maximum of these two options and we are done!

Complexity: time complexity is O(n), because we use dp problem with complexity O(n) twice. 
Space complexity is O(1), because in python lists passed by reference and space complexity of House Robber problem is O(1).
'''
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def rob_helper(nums):       
            dp1 = 0
            dp2 = 0
            for num in nums:
                dp = max(dp2+num,dp1)
                dp2 = dp1
                dp1 = dp          
            return dp1
    
        return max(nums[0] + rob_helper(nums[2:-1]), rob_helper(nums[1:]))

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def rob_helper(nums):
            dp1, dp2 = 0, 0
            for num in nums:
                dp1, dp2 = dp2, max(dp1 + num, dp2)          
            return dp2
    
        return max(nums[0] + rob_helper(nums[2:-1]), rob_helper(nums[1:]))
'''
Bottom up DP - O(1) Space
Since the houses form a circle, so we need to avoid robbing on house[0] and house[n-1] together.
So we divide 2 cases:
    Case 1: Rob the maximum of amount money in houses[0..n-2].
    Case 2: Rob the maximum of amount money in houses[1..n-1].
To solve case 1, case 2, please check this solution 198. House Robber.
Pick the maximum of amount money we can rob in case 1 and case 2.

Complexity:
Time: O(N), where N <= 100 is length of nums array.
Space: O(1)
'''
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def solve(left, right):
            dp, dp1, dp2 = 0, 0, 0
            for i in range(left, right+1):
                dp = max(dp1, dp2 + nums[i])
                dp2 = dp1
                dp1 = dp
            return dp1
        
        n = len(nums)
        if n == 1: return nums[0]
        return max(solve(0, n-2), solve(1, n-1))

# 55. Jump Game
# Solution : Max Pos So Far
# Complexity
# Time: O(N), where N <= 10^4 is length of nums array.
# Space: O(1)
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        maxPos = 0
        i = 0
        while i <= maxPos:
            maxPos = max(maxPos, i + nums[i])
            if maxPos >= n - 1: return True
            i += 1
        return False

# Going forwards. m tells the maximum index we can reach so far.
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        m = 0
        for i, n in enumerate(nums):
            if i > m:
                return False
            m = max(m, i+n)
        return True

# Going backwards
class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        goal = len(nums) - 1
        for i in range(len(nums))[::-1]:
            if i + nums[i] >= goal:
                goal = i
        return not goal

# Day 13 Dynamic Programming
# 45. Jump Game II
'''
Solution : Greedy
The main idea is based on greedy.
Step 1: Let's say the range of the current jump is [left, right], 
farthest is the farthest position that all positions in [left, right] can reach.
Step 2: Once we reach to right, we trigger another jump with left = right + 1, 
right = farthest, then repeat step 1 util we reach at the end.

Complexity
Time: O(N), where N <= 10^4 is the length of array nums
Space: O(1)
'''
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        jumps = 0
        farthest = 0
        left = right = 0
        while right < len(nums) - 1:
            for i in range(left, right + 1):
                farthest = max(farthest, i + nums[i])
            left = right + 1
            right = farthest
            jumps += 1
            
        return jumps
'''
Since each element of our input array (N) represents the maximum jump length and not the definite jump length, 
that means we can visit any index between the current index (i) and i + N[i]. 
Stretching that to its logical conclusion, we can safely iterate through N 
while keeping track of the furthest index reachable (next) at any given moment (next = max(next, i + N[i])). 
We'll know we've found our solution once next reaches or passes the last index (next >= N.length - 1).

The difficulty then lies in keeping track of how many jumps it takes to reach that point. 
We can't simply count the number of times we update next, 
as we may see that happen more than once while still in the current jump's range. 
In fact, we can't be sure of the best next jump until we reach the end of the current jump's range.

So in addition to next, we'll also need to keep track of the current jump's endpoint (curr) 
as well as the number of jumps taken so far (ans).

Since we'll want to return ans at the earliest possibility, we should base it on next, as noted earlier. 
With careful initial definitions for curr and next, 
we can start our iteration at i = 0 and ans = 0 without the need for edge case return expressions.

Time Complexity: O(N) where N is the length of N
Space Cmplexity: O(1)
'''
class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        Nlen, curr, nxt, ans, i = len(nums) - 1, -1, 0, 0, 0
        while nxt < Nlen:
            if i > curr:
                ans += 1
                curr = nxt
            nxt = max(nxt, nums[i] + i)
            i += 1
        return ans

# 62. Unique Paths
# O(n) space 
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        if not m or not n:
            return 0
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] += cur[j-1]
        return cur[-1]

'''
Solution : Math Solution
There are total m+n-2 moves to go from Top-Left to Bottom-Right.
In m+n-2 moves, there are m-1 down moves and n-1 right moves.
You can imagine there are m+n-2 moves as: X X X ... X X X
    X can be one of two values: down D or right R.
    So, basically, it means we need to calculate how many ways we could choose m-1 down moves from m+n-2 moves, or n-1 right moves from m+n-2 moves.
So total ways = C(m+n-2, m-1) = C(m+n-2, n-1) = (m+n-2)! / (m-1)! / (n-1)!.

Complexity
Time: O(M + N), where M <= 100 is number of rows, N <= 100 is number of columns.
Space: O(1)
'''
class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        ans = 1
        j = 1
        for i in range(m, m+n-2 + 1):
            ans *= i
            ans #= j
            j += 1
            
        return ans

# Day 14 Dynamic Programming

# 5. Longest Palindromic Substring

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        length=len(s)

        if(length<=1 or s==s[::-1]):#如果字符串长度为1直接返回
            return s

        start=0
        maxLen=1
        for i in range(0,length):
            ##注意s[1:3]表示的是第2个和第3个char
            if i-maxLen>=1 and s[i-maxLen-1:i+1]==s[i-maxLen-1:i+1][::-1]:
                start=i-maxLen-1
                maxLen+=2
                continue
            if i-maxLen>=0 and s[i-maxLen:i+1]==s[i-maxLen:i+1][::-1]:
                start=i-maxLen
                maxLen+=1

        return s[start:start+maxLen]

# from middle to two ends
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        for i in xrange(len(s)):
            # odd case, like "aba"
            tmp = self.helper(s, i, i)
            if len(tmp) > len(res):
                res = tmp
            # even case, like "abba"
            tmp = self.helper(s, i, i+1)
            if len(tmp) > len(res):
                res = tmp
        return res

    # get the longest palindrome, l, r are the middle indexes   
    # from inner to outer
    def helper(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]

# To make it short, we can use "max":
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        res = ""
        for i in range(len(s)):
            res = max(self.helper(s,i,i), self.helper(s,i,i+1), res, key=len)
        return res
        
    def helper(self,s,l,r):      
        while 0<=l and r < len(s) and s[l]==s[r]:
                l-=1; r+=1
        return s[l+1:r]


# 413. Arithmetic Slices
'''
Solution : Bottom up DP (Space Optimized)
Since our dp only access current dp state dp and previous dp state dpPrev.
So we can easy to achieve O(1) in space.

Complexity
Time: O(N), where N <= 5000 is length of nums array.
Space: O(1)
'''
class Solution(object):
    def numberOfArithmeticSlices(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        dp, dpPrev = 0, 0
        ans = 0
        for i in range(2, n):
            if nums[i-1] - nums[i-2] == nums[i] - nums[i-1]:
                dp = dpPrev + 1
            ans += dp
            dpPrev = dp
            dp = 0
        return ans


# Day 15 Dynamic Programming

# 91. Decode Ways
''' Solution : Bottom-up DP (Space Optimized)
Since our dp only need to keep up to 3 following states:
    Current state, let name dp corresponding to dp[i]
    Last state, let name dp1 corresponding to dp[i+1]
    Last twice state, let name dp2 corresponding to dp[i+2]
So we can achieve O(1) in space.

Complexity
Time: O(N), where N <= 100 is length of string s.
Space: O(1)
'''
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp, dp1, dp2, n = 0, 1, 0, len(s)
        for i in range(n-1, -1, -1):
            if s[i] != '0':  # Single digit
                dp += dp1
            if i+1 < n and (s[i] == '1' or s[i] == '2' and s[i+1] <= '6'):  # Two digits
                dp += dp2
            dp, dp1, dp2 = 0, dp, dp1
        return dp1

# 139. Word Break
'''
The wordDict parameter had been changed to a list of strings (instead of a set of strings).
DP
Time complexity O(n^2)
Space complexity O(n^2)
wordBreak("") && inDict("lettcode")
wordBreak("leet") && inDict("code") <--  
wordBreak("leetcode") && inDict("e)

wordBreak("leet") <-- wordBreak("") && inDict("leet") 

inDict("leet") = true && inDict("code") = true 
wordBeak("") = true
'''
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        def canBreak(s, m, wordDict):
            if s in m: return m[s]
            if s in wordDict: 
                m[s] = True
                return True
            
            for i in range(1, len(s)):
                r = s[i:]
                if r in wordDict and canBreak(s[0:i], m, wordDict):
                    m[s] = True
                    return True
            
            m[s] = False
            return False
            
        return canBreak(s, {}, set(wordDict))

# Day 16 Dynamic Programming

# 300. Longest Increasing Subsequence

'''
Solution : DP + Binary Search / Patience Sorting

Patience Sorting
 It might be easier for you to understand how it works if you think about it as piles of cards instead of tails. 
 The number of piles is the length of the longest subsequence. 

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.
dp is an increasing array, we can use binary search to find the index to insert/update the array.
ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
'''
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        d = []
        for x in nums:
            i = bisect_left(d, x)
            if i == len(d): 
                d.append(x)
            else:
                d[i] = x
        return len(d)

''' Binary search O(nlogn) time

tails is an array storing the smallest tail of all increasing subsequences with length i+1 in tails[i].
For example, say we have nums = [4,5,6,3], then all the available increasing subsequences are:

len = 1   :      [4], [5], [6], [3]   => tails[0] = 3
len = 2   :      [4, 5], [5, 6]       => tails[1] = 5
len = 3   :      [4, 5, 6]            => tails[2] = 6
We can easily prove that tails is a increasing array. Therefore it is possible to do a binary search in tails array to find the one needs update.

Each time we only do one of the two:

(1) if x is larger than all tails, append it, increase the size by 1
(2) if tails[i-1] < x <= tails[i], update tails[i]
Doing so will maintain the tails invariant. The the final answer is just the size.
'''
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        tails = [0] * len(nums)
        size = 0
        for x in nums:
            i, j = 0, size
            while i != j:
                m = (i + j) / 2
                if tails[m] < x:
                    i = m + 1
                else:
                    j = m
            tails[i] = x
            size = max(i + 1, size)
        return size

# 673. Number of Longest Increasing Subsequence
'''
Intuition
To find the frequency of the longest increasing sequence, we need
    First, know how long is the longest increasing sequence
    Second, count the frequency
Thus, we create 2 lists with length n
    dp[i]: meaning length of longest increasing sequence
    cnt[i]: meaning frequency of longest increasing sequence
If dp[i] < dp[j] + 1 meaning we found a longer sequence and dp[i] need to be updated, then cnt[i] need to be updated to cnt[j]
If dp[i] == dp[j] + 1 meaning dp[j] + 1 is one way to reach longest increasing sequence to i, so simple increment by cnt[j] like this cnt[i] = cnt[i] + cnt[j]
Finally, sum up cnt of all longest increase sequence will be the solution
This is a pretty standard DP question. Just like most sequence type of DP question, we need to loop over each element and check all previous stored information to update current.
Time complexity is O(n*n)
'''
class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        n = len(nums)
        m, dp, cnt = 0, [1] * n, [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[i] < dp[j]+1: dp[i], cnt[i] = dp[j]+1, cnt[j]
                    elif dp[i] == dp[j]+1: cnt[i] += cnt[j]
            m = max(m, dp[i])                        
        return sum(c for l, c in zip(dp, cnt) if l == m)

# Simple DP is pretty slow
class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        n = len(nums)
        res, m, dp, cnt = 0, 0, [1] * n, [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[i] < dp[j]+1: dp[i], cnt[i] = dp[j]+1, cnt[j]
                    elif dp[i] == dp[j]+1: cnt[i] += cnt[j]
            if dp[i] == m: res += cnt[i]
            if dp[i] > m: res, m = cnt[i], dp[i]

        return res

# this solution is super slow tho
import numpy
class Solution(object):
    def findNumberOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums: return 0
        n = len(nums)
        res, m, dp, cnt = 0, 0, [1] * n, [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    if dp[i] < dp[j]+1: dp[i], cnt[i] = dp[j]+1, cnt[j]
                    elif dp[i] == dp[j]+1: cnt[i] += cnt[j]
        m = numpy.max(dp) # max of array
        for i in range(n):
            if dp[i] == m: res += cnt[i]
            
        return res

# Day 17 Dynamic Programming

# 1143. Longest Common Subsequence
'''
DP

Use dp[i][j] to represent the length of longest common sub-sequence of text1[0:i] and text2[0:j]
dp[i][j] = dp[i - 1][j - 1] + 1 if text1[i - 1] == text2[j - 1] else max(dp[i][j - 1], dp[i - 1][j])

Time complexity: O(mn)
Space complexity: O(mn) -> O(n)
'''
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        m,n = len(text1),len(text2)    
        dp1,dp2 = [0]*(n+1), [0]*(n+1)     
        for i in range(m):
            for j in range(n) :        
                if text1[i] == text2[j]:
                    dp2[j + 1] = dp1[j] + 1 
                else:
                    dp2[j + 1] = max(dp1[j + 1] , dp2[j])
            dp1,dp2 = dp2,dp1
              
        return dp1[n] 

class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        """
        :type text1: str
        :type text2: str
        :rtype: int
        """
        m,n = len(text1),len(text2)    
        dp = [0]*(n+1)    
        for i in range(m):
            prev = 0; # dp[i][j]
            for j in range(n) :        
                curr = dp[j + 1] # dp[i][j + 1]
                if text1[i] == text2[j]:
                  # dp[i + 1][j + 1] = dp[i][j] + 1
                    dp[j + 1] = prev + 1 
                else:
                  # dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
                    dp[j + 1] = max(curr, dp[j])
                prev = curr
               
        return dp[n] # dp[m][n]

# 583. Delete Operation for Two Strings
'''
Approach 1-D Dynamic Programming [Accepted]:
Complexity Analysis

Time complexity : O(m*n). 
We need to fill in the dpdp array of size nn, mm times. 
Here, mm and nn refer to the lengths of s1s1 and s2s2.

Space complexity : O(n). dp array of size nn is used.
'''
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [0] * (n+1)
        for i in range(0, m+1):
            temp =  [0] * (n+1)
            for j in range(0, n+1):
                if (i == 0 or j == 0 ):
                    temp[j] = i + j
                elif word1[i-1] == word2[j-1]:
                    temp[j] = dp[j-1]
                else: 
                    temp[j] = 1 + min(temp[j-1], dp[j])
            dp = temp
        return dp[n]

# Day 18 Dynamic Programming

# 72. Edit Distance

# Solution : Bottom up DP (Space Optimized)
# Since we build our dp rows by rows, we access only previous dp state dpPrev and current dp state dp.
# So we can optimize to O(N) in Space Complexity.
# Complexity:
# Time: O(M*N), where M <= 500 is length of s1 string, N <= 500 is length of s2 string.
# Space: O(N)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp, dpPrev = [-1] * (n+1), [-1] * (n+1)
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[j] = j  # Need to insert `j` chars to become s2[:j]
                elif j == 0:
                    dp[j] = i  # Need to delete `i` chars to become ""
                elif word1[i-1] == word2[j-1]:
                    dp[j] = dpPrev[j-1]
                else:
                    dp[j] = min(dpPrev[j], dp[j-1], dpPrev[j-1]) + 1
            dp, dpPrev = dpPrev, dp
        return dpPrev[n]

# Solution : Bottom up DP
# Complexity:
# Time: O(M*N), where M <= 500 is length of s1 string, N <= 500 is length of s2 string.
# Space: O(M*N)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [[-1] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j  # Need to insert `j` chars to become word2[:j]
                elif j == 0:
                    dp[i][j] = i  # Need to delete `i` chars to become ""
                elif word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n]

# Iterative DP
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    dp[i][j] = j  # Need to insert `j` chars to become word2[:j]
                elif j == 0:
                    dp[i][j] = i  # Need to delete `i` chars to become ""
                if i >= 1 and j >= 1:    
                    if word1[i-1] == word2[j-1]:
                        c = 0
                    else:
                        c = 1
                    dp[i][j] = min(min(dp[i-1][j], dp[i][j-1])+1, dp[i-1][j-1] + c)
        return dp[m][n]

# Recursive DP super quick
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1, l2 = len(word1), len(word2)
        dp = [[-1] * (l2+1) for _ in range(l1+1)]
        return self.MinDistance(dp,word1,word2,l1,l2)
     
    def MinDistance(self,dp, word1, word2, l1, l2):
        if not l1 : return l2
        if not l2 : return l1
        if dp[l1][l2] >= 0 : return dp[l1][l2]

        ans = 0
        if word1[l1 - 1] == word2[l2 - 1]:
            ans = self.MinDistance(dp, word1, word2, l1 - 1, l2 - 1)
        else :
            ans = min(self.MinDistance(dp, word1, word2, l1 - 1, l2 - 1),
                  min(self.MinDistance(dp, word1, word2, l1 - 1, l2), 
                      self.MinDistance(dp, word1, word2, l1, l2 - 1))) + 1
        dp[l1][l2] = ans  
        return ans    
# Recursive DP similar with above, second quick    
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        l1, l2 = len(word1), len(word2)
        dp = [[-1] * (l2+1) for _ in range(l1+1)]
         
        def minDistance(word1, word2, l1, l2):
            if not l1 : return l2
            if not l2 : return l1
            if dp[l1][l2] >= 0 : return dp[l1][l2]

            ans = 0
            if word1[l1 - 1] == word2[l2 - 1]:
                ans = minDistance(word1, word2, l1 - 1, l2 - 1)
            else :
                ans = min(minDistance(word1, word2, l1 - 1, l2 - 1),
                      min(minDistance(word1, word2, l1 - 1, l2), 
                          minDistance(word1, word2, l1, l2 - 1))) + 1
            dp[l1][l2] = ans  
            return ans    
        
        return minDistance(word1,word2,l1,l2)

# 322. Coin Change

'''
Solution : DP
Time complexity: O(n*amount)
Space complexity: O(amount)
'''
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        INVALID = 2**32
        dp = [INVALID] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                if dp[i - coin] >= dp[i]: continue
                dp[i] = dp[i - coin] + 1
        return -1 if dp[amount] == INVALID else dp[amount]
'''
Solution : DFS+Greedy+Pruning
Use largest and as many as coins first to reduce the search space
Time complexity: O(amount^n/(coin_0*coin_1*…*coin_n))
Space complexity: O(n)
'''
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        coins.sort(reverse=True)
        INVALID = 10**10
        self.ans = INVALID
        def dfs(s, amount, count):      
            if amount == 0:
                self.ans = count
                return
            if s == len(coins): return

            coin = coins[s]
            for k in range(amount # coin, -1, -1):
                if count + k >= self.ans: break
                dfs(s + 1, amount - k * coin, count + k)
        dfs(0, amount, 0)
        return -1 if self.ans == INVALID else self.ans

# 343. Integer Break
class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
    
        if n > 3: n+=1
        dp = [0]*(n+1)
        dp[1] = 1
       # fill the entire dp array
        for i in range(2,n+1):
            for j in range(1,i):
                dp[i] = max(dp[i], j*dp[i - j])

        return dp[n]

class Solution(object):
    def integerBreak(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        dp = [0]*(n+1)
        dp[1] = 1
       # fill the entire dp array
        for i in range(2,n+1):
            for j in range(1,i/2+1):
                dp[i] = max(dp[i],max(j, dp[j]) * max(i - j, dp[i - j]))

        return dp[n]

# Day 19 Bit Manipulation

# 201. Bitwise AND of Numbers Range
"""
The hardest part of this problem is to find the regular pattern.
For example, for number 26 to 30
Their binary form are:
11010
11011
11100
11101
11110
Because we are trying to find bitwise AND, so if any bit there are at least one 0 and one 1, 
it always 0. In this case, it is 11000.
So we are go to cut all these bit that they are different. 
In this case we cut the right 3 bit.
"""
class Solution(object):
    def rangeBitwiseAnd(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: int
        """
        i = 0
        while left != right:
            left >>= 1
            right >>= 1
            i += 1
        return right << i

# Day 20 Others
# 384. Shuffle an Array
# Fisher-Yates Algorithm
'''
The Fisher-Yates algorithm is remarkably similar to the brute force solution. 
On each iteration of the algorithm, we generate a random integer between the current index 
and the last index of the array. 
Then, we swap the elements at the current index and the chosen index - 
this simulates drawing (and removing) the element from the hat, 
as the next range from which we select a random index will not include the most recently processed one. 
One small, yet important detail is that it is possible to swap an element with itself - 
otherwise, some array permutations would be more likely than others. 
'''
# Time Complexity:
# init, reset, shuffle: O(N)
# Space Complexity:
# init: O(N)
# reset: O(N)
# shuffle: O(1)
class Solution(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.backup = nums[:]
        self.nums = nums[:]

    def reset(self):
        """
        :rtype: List[int]
        """
        self.nums = self.backup[:]
        return self.nums
        

    def shuffle(self):
        """
        :rtype: List[int]
        """
        n = len(self.nums)
        for i in range(n):
            j = randint(i, n-1)
            self.nums[i], self.nums[j] = self.nums[j], self.nums[i]
        return self.nums


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.reset()
# param_2 = obj.shuffle()

# Day 21 Others
# 202. Happy Number
# An Alternative Implementation
# Thanks @Manky for sharing this alternative with us!

# This approach was based on the idea that all numbers either end at 1 or 
# enter the cycle {4, 16, 37, 58, 89, 145, 42, 20}, wrapping around it infinitely.

# An alternative approach would be to recognise that all numbers will either end at 1,
# or go past 4 (a member of the cycle) at some point.
# Therefore, instead of hardcoding the entire cycle, we can just hardcode the 4.

class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        def get_next(number):
            total_sum = 0
            while number > 0:
                number, digit = divmod(number, 10)
                total_sum += digit ** 2
            return total_sum
        
        while n != 1 and n != 4:
            n = get_next(n)
            
        return n == 1

"""
Floyd's Cycle-Finding Algorithm 

Intuition
This algorithm is based on 2 runners running around a circular race track, a fast runner and a slow runner.
In reference to a famous fable, many people call the slow runner the "tortoise" and the fast runner the "hare".

Regardless of where the tortoise and hare start in the cycle, they are guaranteed to eventually meet. 
This is because the hare moves one node closer to the tortoise (in their direction of movement) each step.

Algorithm
Instead of keeping track of just one value in the chain, we keep track of 2, called the slow runner and the fast runner. 
At each step of the algorithm, the slow runner goes forward by 1 number in the chain, 
 and the fast runner goes forward by 2 numbers (nested calls to the getNext(n) function).

If n is a happy number, i.e. there is no cycle, then the fast runner will eventually get to 1 before the slow runner.
If n is not a happy number, then eventually the fast runner and the slow runner will be on the same number.

Complexity Analysis
Time complexity : O(log n). Builds on the analysis for the previous approach, 
except this time we need to analyse how much extra work is done by keeping track of two places instead of one, 
and how many times they'll need to go around the cycle before meeting.

If there is no cycle, then the fast runner will get to 1, and the slow runner will get halfway to 1. 
Because there were 2 runners instead of 1, we know that at worst, the cost was O(2 log n) = O(logn).

Like above, we're treating the length of the chain to the cycle as 
insignificant compared to the cost of calculating the next value for the first n. 
Therefore, the only thing we need to do is show that the number of times the runners 
go back over previously seen numbers in the chain is constant.

Once both pointers are in the cycle (which will take constant time to happen) 
the fast runner will get one step closer to the slow runner at each cycle. 
Once the fast runner is one step behind the slow runner, they'll meet on the next step. 
Imagine there are kk numbers in the cycle. If they started at k - 1 places apart
 (which is the furthest apart they can start), 
 then it will take k - 1 steps for the fast runner to reach the slow runner,
  which again is constant for our purposes.
Therefore, the dominating operation is still calculating the next value for the starting n,
 which is O(logn).

Space complexity : O(1). For this approach, we don't need a HashSet to detect the cycles. 
The pointers require constant extra space.
"""
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        def get_next(number):
            total_sum = 0
            while number > 0:
                number, digit = divmod(number, 10)
                total_sum += digit ** 2
            return total_sum

        slow_runner = n
        fast_runner = get_next(n)
        while fast_runner != 1 and slow_runner != fast_runner:
            slow_runner = get_next(slow_runner)
            fast_runner = get_next(get_next(fast_runner))
        return fast_runner == 1

"""
Hardcoding the Only Cycle (Advanced)

Intuition
The previous two approaches are the ones you'd be expected to come up with in an interview. 
This third approach is not something you'd write in an interview, 
but is aimed at the mathematically curious among you as it's quite interesting.

What's the biggest number that could have a next value bigger than itself? 
Well we know it has to be less than 243243, from the analysis we did previously. 
Therefore, we know that any cycles must contain numbers smaller than 243243, 
as anything bigger could not be cycled back to. 
With such small numbers, it's not difficult to write a brute force program that finds all the cycles.

If you do this, you'll find there's only one cycle: 4 => 16 => 37 => 58 => 89 => 145 => 42 => 20 => 44→16→37→58→89→145→42→20→4. 
All other numbers are on chains that lead into this cycle, or on chains that lead into 11.

Therefore, we can just hardcode a HashSet containing these numbers, 
and if we ever reach one of them, then we know we're in the cycle. 
There's no need to keep track of where we've been previously.

Complexity Analysis
Time complexity : O(logn). Same as above.
Space complexity : O(1). We are not maintaining any history of numbers we've seen.
The hardcoded HashSet is of a constant size.
"""
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        cycle_members = {4, 16, 37, 58, 89, 145, 42, 20}

        def get_next(number):
            total_sum = 0
            while number > 0:
                number, digit = divmod(number, 10)
                total_sum += digit ** 2
            return total_sum

        while n != 1 and n not in cycle_members:
            n = get_next(n)

        return n == 1

# 149. Max Points on a Line
# The solution uses slope directly as the key for the dictionary. Python can handle it.
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points) <= 2: return len(points)
        d = collections.defaultdict(int) # slope : count
        result = 0
        for i in range(len(points)):
            d.clear()
            overlap, curmax = 0, 0
            for j in range(i+1, len(points)):
                dx, dy = points[j][0] - points[i][0], points[j][1] - points[i][1]
                if dx == 0 and dy == 0:
                    overlap += 1
                    continue
                slope = dy * 1.0 / dx if dx != 0 else 'infinity' # 1.
                # Use decimal not float for higher precision while dividing
                # slope = Decimal(dy) / dx if dx != 0 else 'infinity' # 2.
                d[slope] += 1
                curmax = max(curmax, d[slope])
            result = max(result, curmax+overlap+1)
        return result

# The second solution uses a tuple of integer (dx, dy) as the key.
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        if len(points) <= 2: return len(points)
        d = collections.defaultdict(int) # (x,y) : count
        result = 0
        for i in range(len(points)):
            d.clear()
            overlap = 0
            curmax = 0
            for j in range(i+1, len(points)):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                if dx == 0 and dy == 0:
                    overlap += 1
                    continue
                gcd = self.getGcd(dx, dy)
                dx #= gcd
                dy #= gcd
                d[(dx,dy)] += 1
                curmax = max(curmax, d[(dx,dy)])
            result = max(result, curmax+overlap+1)
        return result

    def getGcd(self, a, b):
        if b == 0: return a
        return self.getGcd(b, a%b)