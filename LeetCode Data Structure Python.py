# LeetCode Data Structure 
# Day 1 Array
# 217. Contains Duplicate
# Array solution
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """ 
        #using an array
        nums.sort()
        for i in range(0,len(nums)-1):
            if nums[i] == nums[i+1]:
                return True
        return False

class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """ 
        return len(set(nums)) != len(nums) # using set which is much quicker

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        res=Counter(nums) # can also use a counter
        for i in res.values():
            if i>1:
                return True
        return False

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        cont=Counter(nums) # counter version 2 slightly different
        for i in nums:
            if cont[i]>1: 
                return True
        return False

# 53. Maximum Subarray
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        sum, res = nums[0], nums[0]
        for i in range(1, len(nums)):
            sum = nums[i] if sum < 0 else sum + nums[i]
            res = max(res, sum)
        return res

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1, len(nums)):
            if nums[i-1]>0:
                nums[i] += nums[i-1]
        return max(nums)

class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_global = max_current = nums[0]
	
        #subarray ending at position i
        for i in range(1,len(nums)):
            max_current = max(nums[i], nums[i]+max_current)
            max_global = max(max_global,max_current)

        return max(max_global,max_current)

# Day 2 Array
# 1. Two Sum
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {} # dict()
        for i, n in enumerate(nums): 
            if target - n in d:
                return [d[target - n], i]
            else:
                d[n] = i

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = dict()
        for i in range(len(nums)):
        	if target - nums[i] in d:
        		return [d[target - nums[i]], i]
        	else:
        		d[nums[i]] = i

# 88. Merge Sorted Array
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        while n > 0:
            if m < 1 or nums2[n-1] >= nums1[m-1]:  
                nums1[m+n-1] = nums2[n-1]
                n -= 1
            else:
                nums1[m+n-1] = nums1[m-1]
                m -= 1

# Day 3 Array
# 350. Intersection of Two Arrays II
# two-pointer solution
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        n1, n2, res = sorted(nums1), sorted(nums2), []
        p1 = p2 = 0
        while p1 < len(n1) and p2 < len(n2):
            if n1[p1] < n2[p2]:
                p1 += 1
            elif n2[p2] < n1[p1]:
                p2 += 1
            else:
                res.append(n1[p1])
                p1 += 1
                p2 += 1
        return res

# use Counter to make it cleaner
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        counts = collections.Counter(nums1)
        res = []

        for num in nums2:
            if counts[num] > 0:
                res += num,
                counts[num] -= 1

        return res

# another clean way to do it with two Counter's
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        c1, c2 = collections.Counter(nums1), collections.Counter(nums2)
        
        for n in c1:
            if n in c2: res.extend([n]*min(c1[n], c2[n]))
        return res

# 121. Best Time to Buy and Sell Stock
# DP
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices)<=1: 
            return 0
        maxi = 0     #the maximum profit you can get if you sell stock on the current day
        mini = prices[0]  #the minimum price you can buy stock before the current day
        for i in range(len(prices)):
            mini = min(mini,prices[i]) #try to update mini
            maxi = max(maxi,prices[i]-mini) #try to update maxi
        return maxi

# Day 4 Array
# 566. Reshape the Matrix
# numpy reshape() tolist()
import numpy as np
class Solution(object):
    def matrixReshape(self, mat, r, c):
        """
        :type mat: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        try:
            return np.reshape(mat, (r, c)).tolist()
        except:
            return mat

class Solution(object):
    def matrixReshape(self, mat, r, c):
        """
        :type mat: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        m, n = len(mat), len(mat[0])
        if r * c != m * n: return mat  # Invalid size -> return original matrix
        ans = [[0] * c for _ in range(r)]
        for i in range(m * n):
            ans[i # c][i % c] = mat[i # n][i % n]
        return ans
# 118. Pascal's Triangle
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res = [[1]*(i+1) for i in range(numRows)]
        # skip the first two rows because they're just 1's!
        for i in range(2, numRows): 
            for j in range(1, i):
                res[i][j] = res[i-1][j-1] + res[i-1][j]
        return res

'''
Any row can be constructed using the offset sum of the previous row. 
Example:

    1 3 3 1 0 
 +  0 1 3 3 1
 =  1 4 6 4 1
'''
class Solution(object):
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        res = [[1]]
        for i in range(1, numRows):
            res += [map(lambda x, y: x+y, res[-1] + [0], [0] + res[-1])]
            # res here is list of list, so res[-1] gives list
        return res[:numRows] # edge case

# Day 5 Array
# 36. Valid Sudoku
# 5#7 = 0 -> math floor division
class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        rowSet = [set() for r in range(9)]
        colSet = [set() for c in range(9)]
        squareSet = [set() for s in range(9)]

        for r in range(9):
            for c in range(9):
                if board[r][c] != '.':
                    # sr, sc = r # 3, c # 3
                    # sPos = sr * 3 + sc
                    if board[r][c] in rowSet[r] or board[r][c] in colSet[c] or board[r][c] in squareSet[r# 3 * 3  +c # 3]:
                        return False
                    rowSet[r].add(board[r][c])
                    colSet[c].add(board[r][c])
                    squareSet[ r# 3 * 3 +c # 3].add(board[r][c])

        return True

# 74. Search a 2D Matrix
# Solution: Binary Search
# Treat the 2D array as a 1D array. 
# matrix[index / cols][index % cols]
# Time complexity: O(log(m*n))
# Space complexity: O(1)
class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        
        if not matrix :
            return False
        
        l,r = 0,len(matrix) * len(matrix[0])
        
        while l < r:
            m = l+(r-l)#2
            
            if target == matrix[m/len(matrix[0])][m%len(matrix[0])]:
                return True
            elif matrix[m/len(matrix[0])][m%len(matrix[0])] > target:
                r = m
                
            else:
                l = m+1
                
        return False

# Day 6 String
# 387. First Unique Character in a String
# Complexity Analysis
# Time complexity : O(N) since we go through the string of length N two times.
# Space complexity : O(1) because English alphabet contains 26 letters.
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        # build hash map : character and how often it appears
        count = collections.Counter(s)
        
        # find the index
        for i, char in enumerate(s):
            if count[char] == 1:
                return i     
        return -1

# 383. Ransom Note
# O(m+n) with m and n being the lengths of the strings.
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        return (Counter(ransomNote) - Counter(magazine)) == {}
# set(), count(), are both implemented in c.
# collections.Counter is written in python.
class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        for i in set(ransomNote):
            if ransomNote.count(i) > magazine.count(i):
                return False
        return True

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        return all(ransomNote.count(i) <= magazine.count(i) for i in set(ransomNote))       

# 242. Valid Anagram
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return collections.Counter(s) == collections.Counter(t)

class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        return sorted(s) == sorted(t)


# Day 7 Linked List
# 141. Linked List Cycle
# Solution2: Fast + Slow pointers
# Time complexity: O(n) 
# Space complexity: O(1)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        while fast !=None:
            if fast.next == None: return False
            slow = slow.next
            fast = fast.next.next
            if fast == slow: return True
        return False

# 21. Merge Two Sorted Lists
# Solution 2: priority_queue / mergesort
# Recursive O(n)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        if not list1 or not list2:
            return list1 or list2
        if list1.val < list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2

# Solution 1: Iterative O(n)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        dummy = tail = ListNode(0)
        while list1 and list2:
            if list1.val < list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
            
        if not list1 or not list2:
            tail.next = list1 or list2

        return dummy.next

# 203. Remove Linked List Elements
# Solution - II (Recursive)
# Time Complexity : O(N) We are just iterating over the linked list once.
# Space Complexity : O(N), required by implicit recursive stack

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if not head: return None
        head.next = self.removeElements(head.next, val)
        return head.next if head.val == val else head

# Solution - I (Iterative using Dummy node)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = ListNode(-1, head)
        prev = dummy
        while head:
            if head.val != val:
                prev = head
            else:
                prev.next = head.next
            head = head.next
        return dummy.next

# Day 8 Linked List
# 206. Reverse Linked List

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev, curr, nxt = None, head, None;    
        while curr:
            nxt, curr.next = curr.next, prev      
            prev, curr = curr, nxt
        return prev

# 83. Remove Duplicates from Sorted List

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
        cur = head
        while cur:
            while cur.next and cur.next.val == cur.val:
                cur.next = cur.next.next     # skip duplicated node
            cur = cur.next     # not duplicate of current node, move to next node
        return head

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
        if not head: return None
        head.next = self.deleteDuplicates(head.next)
        if not head.next: return head
        return head.next if head.val == head.next.val else head


# Day 9 Stack / Queue
# 20. Valid Parentheses
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        dict = {"]":"[", "}":"{", ")":"("}
        for char in s:
            if char in dict.values():
                stack.append(char)
            elif char in dict.keys():
                if stack == [] or stack.pop() != dict[char] :
                    return False
            else:
                return False
        return stack == []

# 232. Implement Queue using Stacks
class MyQueue(object):

    def __init__(self):
        self.s1 = []
        self.s2 = []
        self.front = None

    def push(self, x):
        """
        :type x: int
        :rtype: None
        """
        if not self.s1: self.front = x
        self.s1.append(x)
        

    def pop(self):
        """
        :rtype: int
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()
        

    def peek(self):
        """
        :rtype: int
        """
        if self.s2 :
            return self.s2[-1]
        return self.front
        

    def empty(self):
        """
        :rtype: bool
        """
        return not self.s1 and not self.s2
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()


# Day 10 Tree
# 144. Binary Tree Preorder Traversal

# Solution 1: Recursion
# Time complexity: O(n)
# Space complexity: O(n)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []
        def preorder(n):
            if not n: return
            ans.append(n.val)
            preorder(n.left)
            preorder(n.right)
        
        preorder(root)
        return ans
        
# Solution 2: Stack
# Time complexity: O(n)
# Space complexity: O(n)

class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans, stack= [], []
        if root : stack.append(root)
        while stack :
            n = stack.pop()
            ans.append(n.val)
            if n.right : stack.append(n.right)
            if n.left : stack.append(n.left)

        return ans

# 94. Binary Tree Inorder Traversal

# Solution: Recursion
# Time complexity: O(n)
# Space complexity: O(h)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []
        self.inorder(root, ans)
        return ans
    
    def inorder(self, node, ans):
        if not node: return
        self.inorder(node.left, ans)
        ans.append(node.val)
        self.inorder(node.right, ans)

# Solution 2: Iterative
# Time complexity: O(n)
# Space complexity: O(h)

class Solution(object):
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root : return []
        ans, s = [], []
        curr = root
        while curr or s :
            while curr :
                s.append(curr)
                curr = curr.left
            
            curr = s.pop()
            ans.append(curr.val)
            curr = curr.right
     
        return ans

# 145. Binary Tree Postorder Traversal

# Solution: Recursion
# Time complexity: O(n)
# Space complexity: O(h)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        ans = []
        def preorder(n):
            if not n: return
            preorder(n.left)
            preorder(n.right)
            ans.append(n.val)
        
        preorder(root)
        return ans

# Solution 2: Iterative
# Time complexity: O(n)
# Space complexity: O(h)

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root: return []
        ans, stack= [], []
        if root : stack.append(root)
        while stack :
            n = stack.pop()
            ans.insert(0, n.val)
            if n.left : stack.append(n.left)
            if n.right : stack.append(n.right)
            
        return ans

class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root : return []
        ans = []
        l = self.postorderTraversal(root.left)
        r = self.postorderTraversal(root.right)

        ans.extend(l)
        ans.extend(r)
        ans.append(root.val)
        
        return ans

# Day 11 Tree
# 102. Binary Tree Level Order Traversal

# Solution 1: BFS O(n)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        ans = []
        curr,nxt = [],[]
        curr.append(root)
        while(curr) :
            ans.append([])
            for node in curr:
                ans[-1].append(node.val)
                if node.left: nxt.append(node.left)
                if node.right: nxt.append(node.right)

            curr,nxt = nxt,curr
            del nxt[:]
        
        return ans

# Solution 2: DFS O(n)

class Solution(object):
    def DFS(self, root,depth,ans):
            if not root: return
            while (len(ans)<=depth):
                ans.append([])
            ans[depth].append(root.val)
            self.DFS(root.left,depth+1,ans)
            self.DFS(root.right,depth+1,ans)
            
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        ans = []
        self.DFS(root,0,ans)
        return ans


# 104. Maximum Depth of Binary Tree

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return 0
        l = self.maxDepth(root.left)
        r = self.maxDepth(root.right)
        return max(l, r) + 1

# 101. Symmetric Tree

class Solution(object):
    def isMirror(self, root1, root2):
          if not root1 and not root2: return True
          if not root1 or not root2: return False
          return root1.val == root2.val \
            and self.isMirror(root1.left, root2.right) \
            and self.isMirror(root2.left, root1.right)
        
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return true
        return self.isMirror(root.left, root.right)

# recursive solutions
class Solution(object):
    def dfs(self, l, r):
        if l and r:
            return l.val == r.val and self.dfs(l.left, r.right) and self.dfs(l.right, r.left)
        return l == r
        
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        return self.dfs(root.left, root.right)

# iterative solutions
def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root:
            return True
        stack = [(root.left, root.right)]
        while stack:
            l, r = stack.pop()
            if not l and not r:
                continue
            if not l or not r or (l.val != r.val):
                return False
            stack.append((l.left, r.right))
            stack.append((l.right, r.left))
        return True

# Day 12 Tree
# 226. Invert Binary Tree
# recursively
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

# BFS

class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        queue = collections.deque([(root)])
        while queue:
            node = queue.popleft()
            if node:
                node.left, node.right = node.right, node.left
                queue.append(node.left)
                queue.append(node.right)
        return root

# DFS
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                node.left, node.right = node.right, node.left
                stack.extend([node.right, node.left])
        return root

# 112. Path Sum 

# Solution: Recursion
# Time complexity: O(n)
# Space complexity: O(n)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def hasPathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: bool
        """
        if not root: return False
        if not root.left and not root.right: return root.val == targetSum
        new_sum = targetSum - root.val
        return self.hasPathSum(root.left, new_sum) or self.hasPathSum(root.right, new_sum) 

# Day 13 Tree
# 700. Search in a Binary Search Tree

# Solution: Recursion
# Time complexity: O(logn ~ n)
# Space complexity: O(logn ~ n)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def searchBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if root == None: return None
        if val == root.val: return root
        elif val > root.val: return self.searchBST(root.right, val)
        return self.searchBST(root.left, val)

# 701. Insert into a Binary Search Tree

# Solution: Recursion
# Time complexity: O(logn ~ n)
# Space complexity: O(logn ~ n)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def insertIntoBST(self, root, val):
        """
        :type root: TreeNode
        :type val: int
        :rtype: TreeNode
        """
        if not root : return TreeNode(val)
        if val > root.val: root.right = self.insertIntoBST(root.right, val)
        else: root.left = self.insertIntoBST(root.left, val)
        return root

# Day 14 Tree
# 98. Validate Binary Search Tree
# Solution 1
# Traverse the tree and limit the range of each subtree and check whether root’s value is in the range.
# Time complexity: O(n)
# Space complexity: O(n)
# Note: in order to cover the range of -2^31 ~ 2^31-1, we need to use long or nullable integer.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.IsValidBST(root, None, None)
  
    def IsValidBST(self, root, minV, maxV) :
            if not root: return True
            if (minV != None and root.val <= minV) or (maxV != None and root.val >= maxV) : return False

            return self.IsValidBST(root.left, minV, root.val) and self.IsValidBST(root.right, root.val, maxV)

# Solution 2
# Do an in-order traversal, the numbers should be sorted,
# thus we only need to compare with the previous number.
# Time complexity: O(n)
# Space complexity: O(n)

class Solution(object):
    def inOrder(self, root): 
            if not root: return True
            if not self.inOrder(root.left): return False
            if self.prev_ != None and root.val <= self.prev_: return False
            self.prev_ = root.val
            return self.inOrder(root.right)
    
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        self.prev_ = None
        return self.inOrder(root)

# 653. Two Sum IV - Input is a BST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        if not root: return False
        bfs, s = [root], set()
        for i in bfs:
            if k - i.val in s: return True
            s.add(i.val)
            if i.left: bfs.append(i.left)
            if i.right: bfs.append(i.right)
        return False

class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        s = set()
        def helper(root, k):
            if not root: return False
            if k - root.val in s: return True
            s.add(root.val)
            return(helper(root.left, k) or helper(root.right, k))
        return helper(root, k)
        
# 235. Lowest Common Ancestor of a Binar

# Solution: Recursion
# Time complexity: O(n)
# Space complexity: O(n)

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if p.val < root.val and q.val < root.val : 
            return self.lowestCommonAncestor(root.left, p, q);
        if p.val > root.val and q.val > root.val :
            return self.lowestCommonAncestor(root.right, p, q);
        return root;

# Day 1 Array
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

# 169. Majority Element

# Approach 3: Sorting
# Time complexity : O(nlgn)O(nlgn)
# Space complexity : O(1) or O(n)

class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        return nums[len(nums)#2]

# Approach 5: Divide and Conquer
# Time complexity : O(nlgn)
# Space complexity : O(lgn)

class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def majority_element_rec(lo, hi):
            # base case; the only element in an array of size 1 is the majority
            # element.
            if lo == hi:
                return nums[lo]

            # recurse on left and right halves of this slice.
            mid = (hi-lo)#2 + lo
            left = majority_element_rec(lo, mid)
            right = majority_element_rec(mid+1, hi)

            # if the two halves agree on the majority element, return it.
            if left == right:
                return left

            # otherwise, count each element and return the "winner".
            left_count = sum(1 for i in range(lo, hi+1) if nums[i] == left)
            right_count = sum(1 for i in range(lo, hi+1) if nums[i] == right)

            return left if left_count > right_count else right

        return majority_element_rec(0, len(nums)-1)

# 15. 3Sum

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
                    l += 1
                    r -= 1
                    while l < r and nums[l] == nums[l-1]:
                        l += 1
                    while l < r and nums[r] == nums[r+1]:
                        r -= 1
                    
        return res

# Day 2 Array
# 75. Sort Colors

class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        red, white, blue = 0, 0, len(nums)-1
    
        while white <= blue:
            if nums[white] == 0:
                nums[red], nums[white] = nums[white], nums[red]
                white += 1
                red += 1
            elif nums[white] == 1:
                white += 1
            else:
                nums[white], nums[blue] = nums[blue], nums[white]
                blue -= 1

# 56. Merge Intervals
# Approach 2: Sorting
# Time complexity : O(nlogn)
# Space complexity : O(logN) or O(n)
class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # if the list of merged intervals is empty or if the current
            # interval does not overlap with the previous, simply append it.
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
            # otherwise, there is overlap, so we merge the current and previous
            # intervals.
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged

# 706. Design HashMap

class MyHashMap(object):

    def __init__(self):
        self.data = [None] * 1000001

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        self.data[key] = value

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        val = self.data[key]
        return val if val != None else -1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        self.data[key] = None


# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

# Day 3 Array
# 119. Pascal's Triangle II

class Solution(object):
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        ans = [1]*(rowIndex+1)
        up = rowIndex
        down = 1
        for i in range(1, rowIndex):
            ans[i] = ans[i-1]*up/down
            up = up - 1
            down = down + 1
        return ans

# 48. Rotate Image
# Approach 1: Rotate Groups of Four Cells
# Time complexity : O(M), as each cell is getting read once and written once.
# Space complexity : O(1) because we do not use any other additional data structures.

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix[0])
        for i in range(n # 2 + n % 2):
            for j in range(n # 2):
                tmp = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1]
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i]
                matrix[j][n - 1 - i] = matrix[i][j]
                matrix[i][j] = tmp

# Approach 2: Reverse on Diagonal and then Reverse Left to Right

# 59. Spiral Matrix II

class Solution(object):
    def generateMatrix(self, n):
        """
        :type n: int
        :rtype: List[List[int]]
        """
        if not n: return []
        res = [[0 for _ in range(n)] for _ in range(n)]
        left, right, top, down, num = 0, n-1, 0, n-1, 1
        while left <= right and top <= down:
            for i in range(left, right+1):
                res[top][i] = num 
                num += 1
            top += 1
            for i in range(top, down+1):
                res[i][right] = num
                num += 1
            right -= 1
            for i in range(right, left-1, -1):
                res[down][i] = num
                num += 1
            down -= 1
            for i in range(down, top-1, -1):
                res[i][left] = num
                num += 1
            left += 1
        return res

# Day 4 Array
# 240. Search a 2D Matrix II

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        c, r = len(matrix[0]) - 1, 0
        while c >= 0 and r < len(matrix):
            if matrix[r][c] > target:
                c -= 1
            elif matrix[r][c] < target:
                r += 1
            else:
                return True
        return False

# 435. Non-overlapping Intervals
'''
further explanation with examples
key concept : pick interval with smallest end, because smallest end can hold most intervals. 
keep track of current element end. 
if next start is more than global end, remove that next element

1. sort by ending.
2. keep track of previous end
3. if the next start > previous end, remove element
example:
arr : [[1,2],[2,3],[3,4],[1,3]]
sorted by end: [[1,2], [2,3], [1,3], [3,4]]

intervals with lowest end will allow us to fit more intervals
if the previous end is more than the next start, remove it.
in this case since [1,3] is removed since 1 is smaller than 3 of the previous end.
because this means that the interval has a smaller start than previous, 
but a bigger end which means that its interval is bigger and hence we should remove it
'''
class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """      
        end, cnt = float('-inf'), 0
        for s, e in sorted(intervals, key=lambda x: x[1]):
            if s >= end: 
                end = e
            else: 
                cnt += 1
        return cnt

class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """      
        start, cnt = float('+inf'), 0
        for s, e in sorted(intervals, key=lambda x: x[0], reverse=True):
            if e <= start: 
                start = s
            else: 
                cnt += 1
        return cnt

# Day 5 Array
# 334. Increasing Triplet Subsequence

class Solution(object):
    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        first = second = float('inf')
        for n in nums:
            if n <= first:
                first = n
            elif n <= second:
                second = n
            else:
                return True
        return False

# 238. Product of Array Except Self
'''
Suppose you have numbers:
Numbers [1      2       3       4       5]
Pass 1: [1  ->  1  ->   12  ->  123  -> 1234]
Pass 2: [2345 <-345 <-  45  <-  5   <-  1]

Finally, you multiply ith element of both the lists to get:
Pass 3: [2345, 1345, 1245, 1235, 1234]
'''
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        p = 1
        n = len(nums)
        output = []
        for i in range(0,n):
            output.append(p)
            p = p * nums[i]
        p = 1
        for i in range(n-1,-1,-1):
            output[i] = output[i] * p
            p = p * nums[i]
        return output

# 560. Subarray Sum Equals K
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        ans, prefsum, d = 0,  0, {0:1}
        for num in nums:
            prefsum += num
            if  prefsum-k  in  d:
                ans = ans + d[prefsum-k]
            d[prefsum] = d.get(prefsum, 0) + 1
        return ans

class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        ans, prefsum,d=0,0,{0:1}

        for num in nums:
            prefsum = prefsum + num

            if prefsum-k in d:
                ans = ans + d[prefsum-k]

            if prefsum not in d:
                d[prefsum] = 1
            else:
                d[prefsum] = d[prefsum]+1

        return ans
# Day 6 String
# 415. Add Strings
class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        ans = []
        i1, i2 = len(num1) - 1, len(num2) - 1
        carry = 0
        while i1 >= 0 or i2 >= 0 or carry > 0:
            if i1 >= 0:
                carry += ord(num1[i1]) - ord('0')
                i1 -= 1
            if i2 >= 0:
                carry += ord(num2[i2]) - ord('0')
                i2 -= 1
            ans.append(chr(carry % 10 + ord('0')))
            carry #= 10
        return "".join(ans)[::-1]
        
# 409. Longest Palindrome
'''
Approach : Greedy [Accepted]

Algorithm

For each letter, say it occurs v times. We know we have v // 2 * 2 letters that can be partnered for sure. 
For example, if we have 'aaaaa', then we could have 'aaaa' partnered, which is 5 // 2 * 2 = 4 letters partnered.

At the end, if there was any v % 2 == 1, then that letter could have been a unique center. 
Otherwise, every letter was partnered. 
To perform this check, we will check for v % 2 == 1 and ans % 2 == 0, 
the latter meaning we haven't yet added a unique center to the answer.

Complexity Analysis
Time Complexity: O(N), where N is the length of s. We need to count each letter.
Space Complexity: O(1), the space for our count, as the alphabet size of s is fixed. 
We should also consider that in a bit complexity model, technically we need O(logN) bits to store the count values.

'''
class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        ans = 0
        for v in collections.Counter(s).itervalues():
            ans += v / 2 * 2
            if ans % 2 == 0 and v % 2 == 1:
                ans += 1
        return ans
        
# Day 7 String
# 290. Word Pattern
'''
At the first glance, the problem can be solved simply by using a hashmap w_to_p which maps words to letters from the pattern. 
But consider this example: w = ['dog', 'cat'] and p = 'aa'. 
In this case, the hashmap doesn't allow us to verify whether we can assign the letter a as a value to the key cat. 
This case can be handled by comparing length of the unique letters from the pattern and unique words from the string.

Space: O(n) - scan
Time: O(n) - for the hashmap
'''
class Solution(object):
    def wordPattern(self, pattern, s):
        """
        :type pattern: str
        :type s: str
        :rtype: bool
        """
        words, w_to_p = s.split(' '), dict()

        if len(pattern) != len(words): return False
        if len(set(pattern)) != len(set(words)): return False # for the case w = ['dog', 'cat'] and p = 'aa'

        for i in range(len(words)):
            if words[i] not in w_to_p: 
                w_to_p[words[i]] = pattern[i]
            elif w_to_p[words[i]] != pattern[i]: 
                return False

        return True

# 763. Partition Labels
'''
Since each letter can appear only in one part, we cannot form a part shorter than 
the index of the last appearance of a letter subtracted by an index of the first appearance. 
For example here (absfab) the lengths of the first part are limited by the positions of the letter a. 
So it's important to know at what index each letter appears in the string last time. 
We can create a hash map and fill it with the last indexes for letters.

Also, we have to validate a candidate part. 
For the same example (absfab) we see that letter a cannot form a border for the first part because of a nasty letter b inside. 
So we need to expand the range of the initial part.

Time: O(n) - 2 sweeps
Space: O(1) - hashmap consist of max 26 keys
'''
class Solution(object):
    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        L = len(s)
        last = {s[i]: i for i in range(L)} # last appearance of the letter
        i, ans = 0, []
        while i < L:
            end, j = last[s[i]], i + 1
            while j < end: # validation of the part [i, end]
                if last[s[j]] > end:
                    end = last[s[j]] # extend the part
                j += 1
           
            ans.append(end - i + 1)
            i = end + 1
            
        return ans

# Day 8 String

# 49. Group Anagrams
'''
Solution 1
Two strings are anagrams if and only if their character counts, 
that is frequencies of each letter a, b, ..., z are the same. 
So it can be done with defauldict(list), 
where key is 26-element list and values are strings, corresponding to this key.

Complexity
Time complexity is O(nk + 26n), where n is number of strings 
and k is the length of the biggest string. Space complexity is O(26n).
'''
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        ans = defaultdict(list)
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord('a')] += 1
            ans[tuple(count)].append(s)
        return ans.values()

'''
Solution 2
First idea is to notice that if we have two anagrams, than when we sort symbols in each of them, 
then we will have exactly the same string. 
So we need for each string to sort it and then use defaultdict.

Complexity
Time complexity will be O(nk * log k), space complexity is O(nk).
'''

class Solution(object): 
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        t = defaultdict(list)
        for s in strs:
            t["".join(sorted(s))].append(s)
        return t.values()

# 43. Multiply Strings

class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        res = [0] * (len(num1)+len(num2))
        for i in range(len(num1)-1, -1, -1):
            carry = 0
            for j in range(len(num2)-1, -1, -1):
                tmp = (ord(num1[i])-ord('0'))*(ord(num2[j])-ord('0')) + carry
                carry = (res[i+j+1]+tmp) # 10
                res[i+j+1] = (res[i+j+1]+tmp) % 10
            res[i] += carry
        res = ''.join(map(str, res))
        return '0' if not res.lstrip('0') else res.lstrip('0')

class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        n, m = len(num1), len(num2)
        if not n or not m:
            return "0"
        
        result = [0] * (n + m)
        for i in reversed(range(n)):
            for j in reversed(range(m)):
                current = int(result[i + j + 1]) + int(num1[i]) * int(num2[j])
                result[i + j + 1] = current % 10
                result[i + j] += current # 10
        
        for i, c in enumerate(result):
            if c != 0:
                return "".join(map(str, result[i:]))
        
        return "0"

# Day 9 String

# 187. Repeated DNA Sequences
# Count occurences of all possible substring with length = 10, there are total 10*(N-9) substrings.
# Just return all substrings which occur more than once.
# Complexity
# Time & Space: O(10*N)
class Solution(object):
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        n = len(s)
        cnt = defaultdict(int)
        for i in range(n - 9):
            cnt[s[i:i+10]] += 1
        
        ans = []
        for key, value in cnt.items():
            if value >= 2: # Found a string that occurs more than once
                ans.append(key)
        return ans

# 5. Longest Palindromic Substring

# Day 10 Linked List

# 2. Add Two Numbers

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = tail = ListNode(0)
        s = 0
        while l1 or l2 or s:
            s+=(l1.val if l1 else 0)+(l2.val if l2 else 0)
            tail.next = ListNode(s % 10)
            tail = tail.next
            s = s //10
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next

# 142. Linked List Cycle II
'''Solution : Fast and Slow

We have 2 phases:
Phase 1: Use Fast and Slow to find the intersection point, 
if intersection point == null then there is no cycle.
Phase 2: Since F = b + m*C, where m >= 0 (see following picture), 
we move head and intersection as the same time, util they meet together, 
the meeting point is the cycle pos.
Complexity:

Time: O(N), where N <= 10^4 is number of elements in the linked list.
Space: O(1)'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head == None or head.next == None: return None

        def findIntersect(head):
            slow = fast = head
            while fast != None and fast.next != None:
                slow = slow.next
                fast = fast.next.next
                if slow == fast:
                    return slow
            return None
        
        # Phase 1: Find the intersection node
        intersect = findIntersect(head)
        if intersect == None: return None
        
        # Phase 2: Find the cycle node
        while head != intersect:
            head = head.next
            intersect = intersect.next
        return head

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        seen = set()
        while head != None:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
        return None

# Day 11 Linked List

# 160. Intersection of Two Linked Lists
'''
In order to solve this problem with only O(1) extra space, 
we'll need to find another way to align the two linked lists. 
More importantly, we need to find a way to line up the ends of the two lists. 
And the easiest way to do that is to concatenate them in opposite orders, A+B and B+A. 
This way, the ends of the two original lists will align on the second half of each merged list.

Then we just need to check if at some point the two merged lists are pointing to the same node. 
In fact, even if the two merged lists don't intersect, 
the value of a and b will be the same (null) when we come to the end of the merged lists, 
so we can use that as our exit condition.

We just need to make sure to string headB onto a and vice versa if one (but not both) list ends.
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a, b = headA, headB
        while (a != b):
            a = headB if not a else a.next
            b = headA if not b else b.next
        return a

# the idea is if you switch head, the possible difference between length would be countered. 
# On the second traversal, they either hit or miss. 
# if they meet, pa or pb would be the node we are looking for, 
# if they didn't meet, they will hit the end at the same iteration, 
# pa == pb == None, return either one of them is the same,None

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        if headA is None or headB is None:
            return None

        a = headA # 2 pointers
        b = headB

        while a is not b:
            # if either pointer hits the end, switch head and continue the second traversal, 
            # if not hit the end, just move on to next
            a = headB if a is None else a.next
            b = headA if b is None else b.next

        return a # only 2 ways to get out of the loop, they meet or the both hit the end=None

# 82. Remove Duplicates from Sorted List II

# Day 12 Linked List

# 24. Swap Nodes in Pairs

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        dummy = ListNode(0)
        dummy.next = head
        head = dummy
        while head.next and head.next.next:
            n1, n2 = head.next, head.next.next
            n1.next = n2.next
            n2.next = n1
            head.next = n2      
            head = n1
        return dummy.next

# 707. Design Linked List

class Node:
    def __init__(self, val, _next=None):
        self.val = val
        self.next = _next
        
class MyLinkedList(object):

    def __init__(self):
        self.head = self.tail = None
        self.size = 0
        
    def getNode(self, index):
        n = Node(0, self.head)
        for i in range(index + 1):
            n = n.next
        return n

    def get(self, index):
        """
        :type index: int
        :rtype: int
        """
        if index < 0 or index >= self.size: return -1
        return self.getNode(index).val

    def addAtHead(self, val):
        """
        :type val: int
        :rtype: None
        """
        n = Node(val, self.head)
        self.head = n
        if self.size == 0:
            self.tail = n
        self.size += 1
        

    def addAtTail(self, val):
        """
        :type val: int
        :rtype: None
        """
        n = Node(val)
        if self.size == 0:
            self.head = self.tail = n
        else:
            self.tail.next = n
            self.tail = n
        self.size += 1

    def addAtIndex(self, index, val):
        """
        :type index: int
        :type val: int
        :rtype: None
        """
        if index < 0 or index > self.size: return
        if index == 0: return self.addAtHead(val)
        if index == self.size: return self.addAtTail(val)
        prev = self.getNode(index - 1)
        n = Node(val, prev.next)
        prev.next = n
        self.size += 1

    def deleteAtIndex(self, index):
        """
        :type index: int
        :rtype: None
        """
        if index < 0 or index >= self.size: return
        prev = self.getNode(index - 1)
        prev.next = prev.next.next
        if index == 0: self.head = prev.next
        if index == self.size - 1: self.tail = prev
        self.size -= 1
        

# Your MyLinkedList object will be instantiated and called as such:
# obj = MyLinkedList()
# param_1 = obj.get(index)
# obj.addAtHead(val)
# obj.addAtTail(val)
# obj.addAtIndex(index,val)
# obj.deleteAtIndex(index)

# Day 13 Linked List

# 25. Reverse Nodes in k-Group
'''
iterative, O(n) time O(1) space
Use a dummy head, and
l, r : define reversing range
pre, cur : used in reversing, standard reverse linked linked list method
jump : used to connect last node in previous k-group to first node in following k-group
'''
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = jump = ListNode(0)
        dummy.next = l = r = head

        while True:
            count = 0
            while r and count < k:   # use r to locate the range
                r = r.next
                count += 1
            if count == k:  # if size k satisfied, reverse the inner linked list
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur  # standard reversing
                    '''
                    temp = cur.next
                    cur.next = pre
                    pre = cur
                    cur = temp
                    '''
                jump.next, jump, l = pre, l, r  # connect two k-groups
                '''
                jump.next = pre
                jump = l
                l = r
                '''
            else:
                return dummy.next

# 143. Reorder List  

# 3-step
# Time  Complexity: O(N)
# Space Complexity: O(1)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return
        
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev, slow.next, slow = None, None, slow.next
        while slow:
            prev, prev.next, slow = slow, prev, slow.next

        slow = head
        while prev:
            slow.next, slow = prev, slow.next
            prev.next, prev = slow, prev.next

# Using List or Queue or Stack 
# Time  Complexity: O(N)
# Space Complexity: O(N)
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        st = []
        cur = head
        while cur != None:
            st.append(cur)
            cur = cur.next
            
        for i in range(len(st) # 2):
            nxt = head.next
            head.next = st.pop()
            head = head.next
            head.next = nxt
            head = head.next
            
        if head != None:
            head.next = None

# Day 14 Stack / Queue

# 155. Min Stack
class MinStack(object):

    def __init__(self):
        self.stack = []

    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.stack.append((val, min(self.getMin(), val))) 
        
    def pop(self):
        """
        :rtype: None
        """
        self.stack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        if self.stack:
            return self.stack[-1][0]

    def getMin(self):
        """
        :rtype: int
        """
        if self.stack:
            return self.stack[-1][1]
        return sys.maxint    

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

# 1249. Minimum Remove to Make Valid Parentheses
'''
Idea
Use stack to remove invalid mismatching parentheses, that is:
Currently, meet closing-parentheses but no opening-parenthesis in the previous -> remove current closing-parenthesis. For example: s = "())".
If there are redundant opening-parenthesis at the end, for example: s = "((()".

Complexity
Time: O(N), where N <= 10^5 is length of string s.
Space: O(N)
'''
class Solution(object):
    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        todoRemove = set()
        openSt = []
        for i, c in enumerate(s):
            if c == '(':
                openSt.append(i)
            elif c == ')':
                if openSt:
                    openSt.pop()
                else:
                    # Meet closing-parentheses but no opening-parenthesis -> remove closing-parenthesis
                    todoRemove.add(i)
        
        for i in openSt:
            todoRemove.add(i)  # remove remain opening-parenthesis

        ans = []
        for i, c in enumerate(s):
            if i not in todoRemove:
                ans.append(c)
        return "".join(ans)

# 1823. Find the Winner of the Circular Game
class Solution(object):
    def findTheWinner(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        res = 0
        for i in range(1, n + 1):
            res = (res + k) % i
        return res + 1

# Day 15 Tree

# 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        def buildBST(l, r):
            if l > r: return None
            m = l + (r - l) // 2
            root = TreeNode(nums[m])
            root.left = buildBST(l, m - 1)
            root.right = buildBST(m + 1, r)
            return root
        return buildBST(0, len(nums) - 1)

# 105. Construct Binary Tree from Preorder and Inorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if inorder:
            ind = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[ind])
            root.left = self.buildTree(preorder, inorder[0:ind])
            root.right = self.buildTree(preorder, inorder[ind+1:])
            return root

# optimized with iterator and map

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        inor_dict = {}
        for i, num in enumerate(inorder):
            inor_dict[num] = i
        pre_iter = iter(preorder)
        
        def helper(start, end):
            if start > end:return None
            root_val = next(pre_iter)
            root = TreeNode(root_val)
            idx = inor_dict[root_val]
            root.left = helper(start, idx-1)
            root.right = helper(idx+1, end)
            return root
        
        return helper(0, len(inorder) - 1)

# optimized Using a dictionary to make things a bit faster

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        inorder_map = {val: i for i, val in enumerate(inorder)}
        return self.dfs_helper(inorder_map, preorder, 0, len(inorder) - 1)

    def dfs_helper(self, inorder_map, preorder, left, right):
        if not preorder : return
        node = preorder.pop(0)
        root = TreeNode(node)
        root_index = inorder_map[node]
        if root_index != left:
            root.left = self.dfs_helper(inorder_map, preorder, left, root_index - 1)
        if root_index != right:
            root.right = self.dfs_helper(inorder_map, preorder, root_index + 1, right)
        return root

# 103. Binary Tree Zigzag Level Order Traversal
'''
In this problem we need to traverse binary tree level by level. When we see levels in binary tree, 
we need to think about bfs, because it is its logic: it first traverse all neighbors, before we go deeper. 
Here we also need to change direction on each level as well. 
So, algorithm is the following:

We create queue, where we first put our root.
result is to keep final result and direction, equal to 1 or -1 is direction of traverse.
Then we start to traverse level by level: if we have k elements in queue currently, we remove them all and put their children instead. 
We continue to do this until our queue is empty. 
Meanwile we form level list and then add it to result, using correct direction and change direction after.
Complexity: time complexity is O(n), where n is number of nodes in our binary tree. 
Space complexity is also O(n), because our result has this size in the end. If we do not count output as additional space, then it will be O(w), where w is width of tree. 
It can be reduces to O(1) I think if we traverse levels in different order directly, but it is just not worth it.
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        queue = deque([root])
        result, direction = [], 1
        
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level[::direction])
            direction *= (-1)
        return result

# Easy-readable efficient

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root: return []
        q = deque([root])
        result = []
        level = 0
        
        while q:
            currentnode = deque()
            levelsize = len(q)
            for _ in range(levelsize):
                node = q.popleft()
                if level % 2 == 0:
                    currentnode.append(node.val)
                else:
                    currentnode.appendleft(node.val)
                    
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            result.append(currentnode)
            level += 1
        return result 

# queue solution

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        deque = collections.deque()
        if root:
            deque.append(root)
        res, level = [], 0
        while deque:
            l, size = [], len(deque)
            for _ in range(size): # process level by level
                node = deque.popleft()
                l.append(node.val)
                if node.left:
                    deque.append(node.left)
                if node.right:
                    deque.append(node.right)
            if level % 2 == 1:
                l.reverse()
            res.append(l)
            level += 1
        return res

# Day 16 Tree

# 199. Binary Tree Right Side View

# DFS
# Complexity
# Time: O(N), where N <= 100 is the nunber of nodes in the binary tree.
# Space: O(H), where H is the height of the binary tree, it's the depth of stack memory.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        self.ans = []
        
        def dfs(root, depth):
            if root == None: return
            if depth == len(self.ans):  # When we meet this `depth` for the first time, let's add the first node as the right side most node.
                self.ans.append(root.val)
            dfs(root.right, depth + 1)  # Go right side first
            dfs(root.left, depth + 1)
            
        dfs(root, 0)
        return self.ans

# BFS - Level by Level
# Complexity
# Time: O(N), where N <= 100 is the nunber of nodes in the binary tree.
# Space: O(N), it's the size of queue in the worst case.

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        # if root == None: return []
        deque = collections.deque()
        if root:
            deque.append(root)
        res = []
        while deque:
            size, val = len(deque), 0
            for _ in range(size):
                node = deque.popleft()
                val = node.val # store last value in each level
                if node.left:
                    deque.append(node.left)
                if node.right:
                    deque.append(node.right)
            res.append(val)
        return res

# 113. Path Sum II

# DFS
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        def dfs(root, targetSum, path):
            if root == None: return None
            
            if root.left == None and root.right == None:  # Is leaf node
                if targetSum == root.val:  # Found a valid path
                    ans.append(list(path)) # list(path) or path[:]
                    ans[-1].append(root.val)
                return
            
            path.append(root.val)
            targetSum -= root.val
            dfs(root.left, targetSum, path)
            dfs(root.right, targetSum, path)
            path.pop()  # backtrack

        ans = []
        dfs(root, targetSum, [])
        return ans

# BFS + queue 
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def pathSum(self, root, targetSum):
        """
        :type root: TreeNode
        :type targetSum: int
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [(root, root.val, [root.val])]
        while queue:
            curr, val, ls = queue.pop(0)
            if not curr.left and not curr.right and val == targetSum:
                res.append(ls)
            if curr.left:
                queue.append((curr.left, val+curr.left.val, ls+[curr.left.val]))
            if curr.right:
                queue.append((curr.right, val+curr.right.val, ls+[curr.right.val]))
        return res

# 450. Delete Node in a BST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if not root: return None
        
        if root.val == key:
            if not root.right: return root.left
            
            if not root.left: return root.right
            
            if root.left and root.right:
                temp = root.right
                while temp.left: temp = temp.left
                root.val = temp.val
                root.right = self.deleteNode(root.right, root.val)

        elif root.val > key:
            root.left = self.deleteNode(root.left, key)
        else:
            root.right = self.deleteNode(root.right, key)
            
        return root

# Day 17 Tree

# 230. Kth Smallest Element in a BST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
    
        return inorder(root)[k - 1]


# 173. Binary Search Tree Iterator
'''
with what we're supposed to support here:

1.    BSTIterator i = new BSTIterator(root);
2.    while (i.hasNext())
3.        doSomethingWith(i.next());
You can see they already have the exact same structure:

Some initialization.
A while-loop with a condition that tells whether there is more.
The loop body gets the next value and does something with it.
So simply put the three parts of that iterative solution into our three iterator methods:
'''
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.root_node=root
        self.current_node=root
        self.stack=[]
        
    def next(self):
        """
        :rtype: int
        """
        while self.current_node:
            self.stack.append(self.current_node)
            self.current_node=self.current_node.left
        next=self.stack.pop()
        self.current_node=next.right
        return next.val     

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.current_node is not None or len(self.stack)!=0

# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class BSTIterator(object):

    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.st = []
        self.pushLeft(root)
       
    def pushLeft(self, root):
        while root != None:
            self.st.append(root)
            root = root.left
        
    def next(self):
        """
        :rtype: int
        """
        node = self.st.pop()
        self.pushLeft(node.right)
        return node.val       

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.st) > 0
        
# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()

# Day 18 Tree
# 236. Lowest Common Ancestor of a Binary Tree

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root == None or root == p or root == q: return root
        # if any((not root, root == p, root == q)): return root # any() only take one arguement
        l = self.lowestCommonAncestor(root.left, p, q)
        r = self.lowestCommonAncestor(root.right, p, q)
        if not l or not r: return l if l else r
        return root

# 297. Serialize and Deserialize Binary Tree
# DFS - Serialize and Deserialize in Pre Order Traversal
# Complexity
# Time: O(N), where N <= 10^4 is number of nodes in the Binary Tree.
# Space: O(N)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        if root == None: return "#"
        return str(root.val) + "," + self.serialize(root.left) + "," + self.serialize(root.right);


    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        nodes = data.split(",")
        self.i = 0

        def dfs():
            if self.i == len(nodes): 
                return None
            nodeVal = nodes[self.i]
            self.i += 1
            if nodeVal == "#": 
                return None
            
            root = TreeNode(int(nodeVal))
            root.left = dfs()
            root.right = dfs()
            return root

        return dfs()

        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# Day 19 Graph

# 997. Find the Town Judge

# Directed Graph
"""
Directed Graph

Intuition:
Consider trust as a graph, all pairs are directed edge.
The point with in-degree - out-degree = N - 1 become the judge.

Explanation:
Count the degree, and check at the end.

Time Complexity:
Time O(T + N), space O(N)
"""
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
"""
Intuition:
Just return the nodes with no in-degres.

Explanation
Quick prove:

Necesssary condition: All nodes with no in-degree must in the final result,
because they can not be reached from
All other nodes can be reached from any other nodes.

Sufficient condition: All other nodes can be reached from some other nodes.

Complexity:
Time O(E)
Space O(N)

in-degree:
number of edges going into a node
If there is no edges coming into a node its a start node and has to be part of the solution set

out-degree:
number of edges coming out of a node

It is important to note that question specifically mentions that the graph is acyclic, 
that's why we are able to simplify the solution and get the job done with just indegree. 
If the graph becomes cyclic then the question can become complicated.
"""
class Solution(object):
    def findSmallestSetOfVertices(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        return list(set(range(n)) - set(j for i, j in edges))

# 841. Keys and Rooms

# iterative DFS
# Time complexity: O(V + E)
# Space complexity: O(V)
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        dfs = [0] # for rooms that we need to visit and we start from room [0]
        seen = set(dfs) # seen -> visited_rooms
        while dfs:
            i = dfs.pop()
            for j in rooms[i]: # j -> key
                if j not in seen: # if key not in visited_rooms
                    dfs.append(j)
                    seen.add(j)
                    if len(seen) == len(rooms): return True
        return len(seen) == len(rooms)

# Recursive DFS
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

# Day 20 Heap (Priority Queue)
# 215. Kth Largest Element in an Array

# Solution : MinHeap
# We use minHeap to keep up to k smallest elements of the nums array.
# Then top of the minHeap is the k largest element.
# Complexity:
# Time: O(NlogK)
# Space: O(K)

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        minHeap = []
        for x in nums:
            heappush(minHeap, x)
            if len(minHeap) > k:
                heappop(minHeap)
        return minHeap[0]

# Solution 2: MaxHeap
# Complexity:
# Time: O(N + KlogN), heapify cost O(N), heappop k times costs O(KlogN).
# Space: O(N)
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        maxHeap = [-x for x in nums]
        heapify(maxHeap)
        for i in range(k-1):
            heappop(maxHeap)
        return -maxHeap[0]
# Solution : Quick Select
# Complexity:
# Time: O(N) in the avarage case, O(N^2) in the worst case. Worst case happens when:
# k = len(nums) and pivot is always the smallest element, 
# so it divides array by [zero elements in the small, 1 element in the equal, n-1 elements in the large], 
# so it always goes to the right side with n-1 elements each time.
# k = 1 and pivot is always the largest element, so it divides array by [n-1 elements in the small, 
# 1 element in the equal, zero elements in the large], 
# so it always goes to the left side with n-1 elements reach time.
# Space: O(N)

class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.findKthSmallest(nums, len(nums) - k + 1)

    def findKthSmallest(self, nums, k):
        if len(nums) <= 1: return nums[0]
        pivot = random.choice(nums)
        small = [x for x in nums if x < pivot]
        equal = [x for x in nums if x == pivot]
        large = [x for x in nums if x > pivot]
        if k <= len(small):
            return self.findKthSmallest(small, k)
        if k <= len(small) + len(equal):
            return pivot
        return self.findKthSmallest(large, k - len(small) - len(equal))
# Solution : Quick Select (Use In-Space memory)
# Complexity:
# Time: O(N) in the avarage case, O(N^2) in the worst case.
# Space: O(N), recursion depth can up to O(N).
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.findKthSmallest(nums, 0, len(nums) - 1, len(nums) - k + 1 - 1)

    def findKthSmallest(self, nums, left, right, k):  # k is one-base indexing
        def partition(left, right, pivotIndex):
            pivot = nums[pivotIndex]
            
            # Move pivot to the right most
            nums[right], nums[pivotIndex] = nums[pivotIndex], nums[right]
            pivotIndex = left
            
            # Swap elements less than pivot to the left
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[pivotIndex], nums[i] = nums[i], nums[pivotIndex]
                    pivotIndex += 1
                    
            # Move pivot to the right place
            nums[pivotIndex], nums[right] = nums[right], nums[pivotIndex]
            return pivotIndex
        
        if left == right:
            return nums[left]
        
        pivotIndex = random.randint(left, right)  # Rand between [left, right]
        pivotIndex = partition(left, right, pivotIndex)
        if pivotIndex == k:
            return nums[pivotIndex]
        if k < pivotIndex:
            return self.findKthSmallest(nums, left, pivotIndex - 1, k)
        return self.findKthSmallest(nums, pivotIndex + 1, right, k)

# 347. Top K Frequent Elements
# Solution : Bucket Sort
# Since the array nums has size of n, the frequency can be up to n.
# We can create bucket to store numbers by frequency.
# Then start bucketIdx = n, we can get the k numbers which have largest frequency.
# Complexity
# Time: O(N), where N <= 10^5 is length of nums array.
# Space: O(N)
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        cnt = Counter(nums)
        
        n = len(nums)
        bucket = [[] for _ in range(n+1)]
        for num, freq in cnt.items():
            bucket[freq].append(num)
            
        bucketIdx = n
        ans = []
        while k > 0:
            while not bucket[bucketIdx]:  # Skip empty bucket
                bucketIdx -= 1
                
            for num in bucket[bucketIdx]:
                if k == 0: break
                ans.append(num)
                k -= 1
            bucketIdx -= 1
        return ans

# Day 21 Heap (Priority Queue)
# 451. Sort Characters By Frequency

# Solution : Counter & Bucket Sort
# Since freq values are in range [0...n],
# so we can use Bucket Sort to achieve O(N) in Time Complexity.
# Complexity
# Time: O(N), where N <= 5 * 10^5 is the length of string s.
# Space: O(N)
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        cnt = Counter(s)
        n = len(s)
        bucket = [[] for _ in range(n+1)]
        for c, freq in cnt.items():
            bucket[freq].append(c)
        
        ans = []
        for freq in range(n, -1, -1):
            for c in bucket[freq]:
                ans.append(c * freq)
        return "".join(ans)

# Solution : Counter & Sorting String S
# Complexity
# Time: O(NlogN), where N <= 5 * 10^5 is the length of string s.
# Space:
# C++: O(logN) it's the stack memory of introsort in C++.
# Python: O(N)
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        cnt = Counter(s)
        s = list(s)
        s.sort(key=lambda x:(-cnt[x], x))
        return "".join(s)

# HashTable + Sort 
# Time Complexity: O(nlogn)
# Space Complexity: O(n)
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """       
        # Count the occurence on each character
        cnt = collections.defaultdict(int)
        for c in s:
            cnt[c] += 1

        # Sort and Build string
        res = []
        for k, v in sorted(cnt.items(), key = lambda x: -x[1]):
            res += [k] * v
        return "".join(res)

# Note that we can optimize the first solution to O(n) by using Counter() in Python
# Time Complexity: O(nlogk), we don't need to sort here, 
# the most_common() cost O(nlogk) based on source code.
# In fact, the most_common used heapq on th implementation.
# Thus, we can consider this solution is the same as solution 2.
# Space Complexity: O(n)
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        # Count the occurence on each character
        cnt = collections.Counter(s)

        # Build string
        res = []
        for k, v in cnt.most_common():
            res += [k] * v
        return "".join(res)

# HashTable + Heap
# Time Complexity: O(nlogk), where k is the number of distinct character.
# Space Complexity: O(n)
# Remember that if you want to achieve Heap structure in Python, you can use heapq and PriorityQueue in Java
class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """      
        # Count the occurence on each character
        cnt = collections.Counter(s)

        # Build heap
        heap = [(-v, k) for k, v in cnt.items()]
        heapq.heapify(heap)

        # Build string
        res = []
        while heap:
            v, k = heapq.heappop(heap)
            res += [k] * -v
        return ''.join(res)

# 973. K Closest Points to Origin

# Sort using heap of size K, O(NlogK)
class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        return heapq.nsmallest(k, points, lambda (x, y): x * x + y * y)

# Max Heap
# Same problem: 215. Kth Largest Element in an Array.
# We use maxHeap to keep k smallest elements in array n elements.
# In the max heap, the top is the max element of the heap and it costs in O(1) in time complexity.
# By using max heap, we can remove (n-k) largest elements and keep k smallest elements in array.
# Complexity
# Time: O(NlogK), where N <= 10^4 is number of points.
# Extra Space (don't count output as space): O(K), it's size of maxHeap.
class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        maxHeap = []
        for x, y in points:
            heappush(maxHeap, [-x*x-y*y, x, y])  # default is minHeap, put negative sign to reverse order
            if len(maxHeap) > k:
                heappop(maxHeap)
        
        return [[x, y] for _, x, y in maxHeap]

# Min Heap
# Same problem: 215. Kth Largest Element in an Array.
# Complexity
# Time: O(N + KlogN), where N <= 10^4 is number of points.
# Extra Space (don't count output as space): O(N).
class Solution(object):
    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        minHeap = []
        for x, y in points:
            minHeap.append([x*x + y*y, x, y])
            
        heapify(minHeap)
        ans = []
        for i in range(k):
            _, x, y = heappop(minHeap)
            ans.append([x, y])
        return ans