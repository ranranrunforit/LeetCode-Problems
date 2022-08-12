// Day 1 Prefix Sum
// 1480. Running Sum of 1d Array
/* Prefix Sum
Explanation
Let B[i] = A[0] + A[1] + .. + A[i]
B[i] = B[i-1] + A[i]


Complexity
Time O(N)
Space O(N)
Space O(1) if changing the input, like in Java.
*/
public class Solution {
    public int[] RunningSum(int[] nums) {
        for (int i = 1; i < nums.Length; ++i)
            nums[i] += nums[i - 1];
        return nums;
    }
}
// 724. Find Pivot Index
// DP
public class Solution {
    public int PivotIndex(int[] nums) {
        // int sum = nums.Sum(); works too!
        int sum = 0;
        Array.ForEach(nums, i => sum += i);
        int l = 0;
        int r = sum;
        for (int i = 0; i < nums.Length; ++i) {
            r -= nums[i];
            if (l == r) return i;
            l += nums[i];
        }
        return -1;
    }
}
/*
Approach #: Prefix Sum
Complexity Analysis

Time Complexity: O(N), where NN is the length of nums.
Space Complexity: O(1), the space used by leftsum and S.
*/
public class Solution {
    public int PivotIndex(int[] nums) {
        int sum = 0, leftsum = 0;
        foreach (int x in nums) sum += x;
        for (int i = 0; i < nums.Length; ++i) {
            if (leftsum == sum - leftsum - nums[i]) return i;
            leftsum += nums[i];
        }
        return -1;
    }
}
// Day 2 String
// 205. Isomorphic Strings
public class Solution {
    public bool IsIsomorphic(string s, string t) {
        int n = s.Length;
        HashSet<int> s1 = new HashSet<int>();    
    for (int i = 0; i < n; ++i){
        s1.Add((s[i] << 8) | t[i]);  
        
    }
          
    HashSet<char> s2 = s.ToHashSet();
    HashSet<char> s3 = new HashSet<char>(t.ToCharArray());
    return s1.Count == s2.Count && s2.Count == s3.Count;
    }
}
// 392. Is Subsequence

// Day 3 Linked List
// 21. Merge Two Sorted Lists

// 206. Reverse Linked List

// Day 4 Linked List
// 876. Middle of the Linked List

// 142. Linked List Cycle II

// Day 5 Greedy
// 121. Best Time to Buy and Sell Stock

// 409. Longest Palindrome

// Day 6 Tree
// 589. N-ary Tree Preorder Traversal

// 102. Binary Tree Level Order Traversal

// Day 7 Binary Search
// 704. Binary Search

// 278. First Bad Version

// Day 8 Binary Search Tree
// 98. Validate Binary Search Tree

// 235. Lowest Common Ancestor of a Binary Search Tree

// Day 9 Graph/BFS/DFS
// 733. Flood Fill

// 200. Number of Islands

// Day 10 Dynamic Programming
// 509. Fibonacci Number

// 70. Climbing Stairs

// Day 11 Dynamic Programming
// 746. Min Cost Climbing Stairs

// 62. Unique Paths

// Day 12 Sliding Window/Two Pointer
// 438. Find All Anagrams in a String

// 424. Longest Repeating Character Replacement
/* Sliding Window, just O(n)
Explanation
maxf means the max frequency of the same character in the sliding window.
To better understand the solution,
you can firstly replace maxf with max(count.values()),
Now I improve from O(26n) to O(n) using a just variable maxf.


Complexity
Time O(n)
Space O(128)
*/
public class Solution {
    public int CharacterReplacement(string s, int k) {
         int res = 0, maxf = 0; int[] count = new int[128];
        for (int i = 0; i < s.Length; ++i) {
            maxf = Math.Max(maxf, ++count[s[i]]);
            if (res - maxf < k)
                res++;
            else
                count[s[i - res]]--;
        }
        return res;
    }
}
/*
Solution 2
Another version of same idea.
In a more standard format of sliding window.
Maybe easier to understand

Time O(N)
Space O(26)
*/
public class Solution {
    public int CharacterReplacement(string s, int k) {
         int maxf = 0, i = 0, n = s.Length;int[] count = new int[26];
        for (int j = 0; j < n; ++j) {
            maxf = Math.Max(maxf, ++count[s[j] - 'A']);
            if (j - i + 1 > maxf + k)
                --count[s[i++] - 'A'];
        }
        return n - i;
    }
}

// Day 13 Hashmap
// 1. Two Sum

// 299. Bulls and Cows
/*The idea is to iterate over the numbers in secret and in guess and count all bulls right away. 
For cows maintain an array that stores count of the number appearances in secret and in guess. 
Increment cows when either number from secret was already seen in guest or vice versa.
*/
public class Solution {
    public string GetHint(string secret, string guess) {
        int bulls = 0;
    int cows = 0;
    int[] numbers = new int[10];
    for (int i = 0; i<secret.Length; i++) {
        if (secret[i] == guess[i]) bulls++;
        else {
            if (numbers[secret[i]-'0']++ < 0) cows++;
            if (numbers[guess[i]-'0']-- > 0) cows++;
        }
    }
    return bulls + "A" + cows + "B";
    }
}
// Day 14 Stack
// 844. Backspace String Compare

// 394. Decode String
/*
Solution 1: Recursion
Time complexity: O(n^2)
Space complexity: O(n)
*/
public class Solution {
    public string DecodeString(string s) {
        if (s.Length == 0) return "";    
        string ans = String.Empty;
        int i = 0;
        int n = s.Length;
        int c = 0;
        while (Char.IsNumber(s[i]) && i < n) 
          c = c * 10 + (s[i++] - '0');

        int j = i;
        if (i < n && s[i] == '[') {      
          int open = 1;
          while (++j < n && open > 0) {
            if (s[j] == '[') ++open;
            if (s[j] == ']') --open;
          }
        } else {
          while (j < n && Char.IsLetter(s[j])) ++j;
        }    

        // "def2[ac]" => "def" + decodeString("2[ac]")
        //  |  |
        //  i  j
        if (i == 0)
          return s.Substring(0, j) + DecodeString(s.Substring(j));

        // "3[a2[c]]ef", ss = decodeString("a2[c]") = "acc"
        //   |      |
        //   i      j
        string ss = DecodeString(s.Substring(i + 1, j - i - 2));    
        while (c-- > 0){
            ans += ss;
        }

        // "3[a2[c]]ef", ans = "abcabcabc", ans += decodeString("ef")
        ans += DecodeString(s.Substring(j));
        return ans;
    }
}
// Day 15 Heap
// 1046. Last Stone Weight
/* Priority Queue
Explanation
Put all elements into a priority queue.
Pop out the two biggest, push back the difference,
until there are no more two elements left.


Complexity
Time O(NlogN)
Space O(N)
*/
public class Solution {
    public int LastStoneWeight(int[] stones) {
        PriorityQueue<int,int> pq = new PriorityQueue<int,int>(Comparer<int>.Create((a, b)=> b - a));
        foreach (int a in stones)
            pq.Enqueue(a, a);
        while (pq.Count > 1){
            pq.TryDequeue(out int item, out int priority);
            pq.TryDequeue(out int item1, out int priority1);

            pq.Enqueue(item - item1 , item - item1);
        }
            
         pq.TryDequeue(out int item2, out int priority2);
        return item2;
    }
}
// 692. Top K Frequent Words
// fold projection into GroupBy and swap Take with Select like this :
public class Solution {
    public IList<string> TopKFrequent(string[] words, int k) {
        return words.GroupBy(w => w, (word, group) => new { Word = word, Count = group.Count() })
                    .OrderByDescending(item => item.Count)
                    .ThenBy(item => item.Word)
                    .Take(k)
                    .Select(item => item.Word)
                    .ToList();
    }
}

public class Solution {
    public IList<string> TopKFrequent(string[] words, int k) {
        return words.GroupBy(x => x)
                .Select(x => new { word = x.Key, count = x.Count() })
                .OrderByDescending(x => x.count)
                .ThenBy(x => x.word)
                .Select(x => x.word)
                .Take(k)
                .ToList();
    }
}

// Day 1 Implementation/Simulation
// 202. Happy Number

// 54. Spiral Matrix

// 1706. Where Will the Ball Fall
/*
Explanation
We drop the ball at grid[0][i]
and track ball position i1, which initlized as i.

An observation is that,
if the ball wants to move from i1 to i2,
we must have the board direction grid[j][i1] == grid[j][i2]


Complexity
Time O(mn)
Space O(n)
*/
public class Solution {
    public int[] FindBall(int[][] grid) {
        int m = grid.Length, n = grid[0].Length; int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int i1 = i, i2;
            for (int j = 0; j < m; ++j) {
                i2 = i1 + grid[j][i1];
                if (i2 < 0 || i2 >= n || grid[j][i2] != grid[j][i1]) {
                    i1 = -1;
                    break;
                }
                i1 = i2;
            }
            res[i] = i1;
        }
        return res;
    }
}
// Day 2 String
// 14. Longest Common Prefix
public class Solution {
    public string LongestCommonPrefix(string[] strs) {
        if (strs == null || strs.Length == 0)
            return "";
        
        Array.Sort(strs);
        String first = strs[0];
        String last = strs[strs.Length - 1];
        int c = 0;
        while(c < first.Length)
        {
            if (first[c] == last[c])
                c++;
            else
                break;
        }
        return c == 0 ? "" : first.Substring(0, c);
    }
}

public class Solution {
    public string LongestCommonPrefix(string[] strs) {
        if(strs == null || strs.Length == 0)    return "";
        String pre = strs[0];
        int i = 1;
        while(i < strs.Length){
            while(strs[i].IndexOf(pre) != 0)
                pre = pre.Substring(0,pre.Length-1);
            i++;
        }
        return pre;
    }
}

public class Solution {
    public string LongestCommonPrefix(string[] strs) {
        if (strs.Length == 0) return "";
        String pre = strs[0];
        for (int i = 1; i < strs.Length; i++)
            while(strs[i].IndexOf(pre) != 0)
                pre = pre.Substring(0,pre.Length-1);
        return pre;
    }
}
// 43. Multiply Strings

// Day 3 Linked List
// 19. Remove Nth Node From End of List

// 234. Palindrome Linked List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public bool IsPalindrome(ListNode head) {
        ListNode slow = head, fast = head, prev, temp;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        prev = slow;
        slow = slow.next;
        prev.next = null;
        while (slow != null) {
            temp = slow.next;
            slow.next = prev;
            prev = slow;
            slow = temp;
        }
        fast = head;
        slow = prev;
        while (slow != null) {
            if (fast.val != slow.val) return false;
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }
}
/*
Example : 1-> 2-> 3-> 4-> 2-> 1

ref points 1 initially.
Make recursive calls until you reach the last element - 1.
On returning from each recurssion, check if it is equal to ref values. 
ref values are updated to on each recurssion.
So first check is ref 1 -  end 1
Second ref 2 - second last 2 ...and so on.*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    private ListNode re;
    public bool IsPalindrome(ListNode head) {
        re = head;        
        return check(head);
    }
    
    public bool check(ListNode node){
        if(node == null) return true;
        bool ans = check(node.next);
        bool isEqual = (re.val == node.val)? true : false; 
        re = re.next;
        return ans && isEqual;
    }
}
// Day 4 Linked List
// 328. Odd Even Linked List
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode OddEvenList(ListNode head) {
         if (head == null || head.next == null) {
        return head;
    }
    ListNode p1 = head, p2 = head.next, pre = p2;
    while (p2 != null && p2.next != null) {
        p1.next = p2.next;
        p1 = p1.next;
        p2.next = p1.next;
        p2 = p2.next;
    }
    p1.next = pre;
    return head;
    }
}
// 148. Sort List
/*
Solution: Merge Sort
Top-down (recursion)

Time complexity: O(nlogn)
Space complexity: O(logn)*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int val=0, ListNode next=null) {
 *         this.val = val;
 *         this.next = next;
 *     }
 * }
 */
public class Solution {
    public ListNode SortList(ListNode head) {
        if (head == null || head.next == null) return head;
    ListNode slow = head;
    ListNode fast = head.next;
    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    ListNode mid = slow.next;
    slow.next = null;
    return merge(SortList(head), SortList(mid));
  }
    private ListNode merge(ListNode l1, ListNode l2) {
    ListNode dummy = new ListNode(0);
    ListNode tail = dummy;
    while (l1 != null && l2 != null) {
      if (l1.val > l2.val) {
        ListNode tmp = l1;
        l1 = l2;
        l2 = tmp;
      }
      tail.next = l1;
      l1 = l1.next;
      tail = tail.next;
    }
    tail.next = (l1 != null) ? l1 : l2;
    return dummy.next;
  }
}
// Day 5 Greedy
// 2131. Longest Palindrome by Concatenating Two Letter Words
/*
Explanation:

2 letter words can be of 2 types:

Where both letters are same
Where both letters are different
Based on the above information:

If we are able to find the mirror of a word, ans += 4
The variable unpaired is used to store the number of unpaired words 
with both letters same.
Unpaired here means a word that has not found its mirror word.
At the end if unpaired same letter words are > 0, 
we can use one of them as the center of the palindromic string.
*/
public class Solution {
    public int LongestPalindrome(string[] words) {
        int[][] counter = new int[26][];
        for (int i = 0 ; i< 26; i++){
            counter[i] = new int[26];
        }
    int ans = 0;
    foreach (String w in words) {
        int a = w[0] - 'a', b = w[1] - 'a';
        if (counter[b][a] > 0) {
            ans += 4; 
            counter[b][a]--; 
        }
        else counter[a][b]++;
    }
    for (int i = 0; i < 26; i++) {
        if (counter[i][i] > 0) {
            ans += 2;
            break;
        }
    }
    return ans;
    }
}
// 621. Task Scheduler
/* PriorityQueue and HashMap
Greedy - It's obvious that we should always process the 
task which has largest amount left.
Put tasks (only their counts are enough, 
we don't care they are 'A' or 'B') in a PriorityQueue in descending order.
Start to process tasks from front of the queue. 
If amount left > 0, put it into a coolDown HashMap
If there's task which cool-down expired, 
put it into the queue and wait to be processed.
Repeat step 3, 4 till there is no task left.
*/
public class Solution {
    public int LeastInterval(char[] tasks, int n) {
        if (n == 0) return tasks.Length;
        
        Dictionary<char, int> taskToCount = new Dictionary<char, int>();
        foreach (char c in tasks) {
            taskToCount[c] = taskToCount.GetValueOrDefault(c, 0) + 1;
        }
        
        PriorityQueue<int,int> queue = new PriorityQueue<int,int>(Comparer<int>.Create((i1, i2) => i2 - i1));
        foreach (char c in taskToCount.Keys) {
            queue.Enqueue(taskToCount[c],taskToCount[c]);
        }
        
        Dictionary<int, int> coolDown = new Dictionary<int, int>();
        int currTime = 0;
        while (queue.Count != 0 || coolDown.Count != 0 ) {
            if (coolDown.ContainsKey(currTime - n - 1)) {
                int t = coolDown[currTime - n - 1];
                coolDown.Remove(currTime - n - 1);
                queue.Enqueue(t,t);
            }
            if (queue.Count != 0) {
                int left = queue.Dequeue() - 1;
        	if (left != 0) coolDown[currTime] =  left;
            }
            currTime++;
        }
        
        return currTime;
    }
}

// Day 6 Tree
// 226. Invert Binary Tree

// 110. Balanced Binary Tree

// Day 7 Tree
// 543. Diameter of Binary Tree
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
     int max = 0;
    public int DiameterOfBinaryTree(TreeNode root) {
        maxDepth(root);
        return max;
        //For every node, length of longest path which pass it = MaxDepth of 
        //its left subtree + MaxDepth of its right subtree.
    }
    private int maxDepth(TreeNode root) {
        if (root == null) return 0;
        
        int left = maxDepth(root.left);
        int right = maxDepth(root.right);
        
        max = Math.Max(max, left + right);
        
        return Math.Max(left, right) + 1;
    }
}
/* Iterative
The idea is to use Post order traversal which means make sure the node is there 
till the left and right childs are processed that's the reason you use peek method 
in the stack to not pop it off without being done with the left and right child nodes. 
Then for each node calculate the max of the left and right sub trees depth 
and also simultaneouslt caluculate the overall max of the left and right subtrees count.
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int DiameterOfBinaryTree(TreeNode root) {
        if(root == null){
            return 0;
        }
        int ans = 0;
        Stack<TreeNode> s = new Stack<TreeNode>();
        Dictionary<TreeNode,int> d = new Dictionary<TreeNode,int>();
        s.Push(root);
        while(s.Count!= 0){
            TreeNode node = s.Peek();
           int left = 0;int right = 0 ;
            if(node.left != null && !d.ContainsKey(node.left)){
                s.Push(node.left);
            }else if(node.right!=null && !d.ContainsKey(node.right)){
                s.Push(node.right);
            }
           else {
                TreeNode temp = s.Pop();
                //dict.ContainsKey(key) ? dict[Key] : defaultValue
                // children's results will never be used again, safe to delete here.
                if (temp.left != null){
                     left =  d[temp.left] ;
                     d.Remove(temp.left);}
                
                if (temp.right != null){
                    right =  d[temp.right] ;
                    d.Remove(temp.right);}
                
                d[temp] = 1 + Math.Max(left,right);
                ans = Math.Max(ans,left + right );
            }
            
        }
        return ans;
    }
}

// 437. Path Sum III
/*
Solution 2: Running Prefix Sum
Same idea to 花花酱 LeetCode 560. Subarray Sum Equals K

Time complexity: O(n)
Space complexity: O(h)
*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int PathSum(TreeNode root, int targetSum) {
        ans_ = 0;
        sums_ = new Dictionary<long, int>(){{(long)0, 1}};
        pathSum(root, 0, (long)targetSum);
        return ans_;
  }
    private int ans_;
    private Dictionary<long, int> sums_;
  
  void pathSum(TreeNode root, long cur, long sum) {
    if (root == null) return;
    cur += root.val;
    ans_ += sums_.GetValueOrDefault(cur - sum , 0);
    sums_[cur] = sums_.GetValueOrDefault(cur, 0)+1;
    pathSum(root.left, cur, sum);
    pathSum(root.right, cur, sum);
    --sums_[cur];
    }
}

/*Solution 1: Recursion
Time complexity: O(n^2)
Space complexity: O(n)*/   
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public int PathSum(TreeNode root, int targetSum) {
       if (root == null) return 0;
    return numberOfPaths(root, (long)targetSum) + PathSum(root.left, targetSum) + PathSum(root.right, targetSum);
  }
private int numberOfPaths(TreeNode root, long left) {
    if (root == null) return 0;    
    left -= root.val;
    return (left == 0 ? 1 : 0) + numberOfPaths(root.left, left) + numberOfPaths(root.right, left);
    }
}
// Day 8 Binary Search
// 74. Search a 2D Matrix

// 33. Search in Rotated Sorted Array

// Day 9 Binary Search Tree
// 108. Convert Sorted Array to Binary Search Tree

// 230. Kth Smallest Element in a BST

// 173. Binary Search Tree Iterator

// Day 10 Graph/BFS/DFS
// 994. Rotting Oranges

// 417. Pacific Atlantic Water Flow

// Day 11 Graph/BFS/DFS
// 210. Course Schedule II
public class Solution {
    public int[] FindOrder(int numCourses, int[][] prerequisites) {
        List<List<int>> graph = new List<List<int>>();
        
        for (int i = 0; i < numCourses; ++i)
            graph.Add(new List<int>());
        
        for (int i = 0; i < prerequisites.Length; ++i) {
            int course = prerequisites[i][0];
            int prerequisite = prerequisites[i][1];            
            graph[course].Add(prerequisite);
        }
        
        int[] visited = new int[numCourses];
        List<int> ans = new List<int>();
        int index = numCourses;
        for (int i = 0; i < numCourses; ++i)
            if (dfs(i, graph, visited, ans)) return new int[0];        
        
        return ans.ToArray();
    }
    
    private bool dfs(int curr, List<List<int>> graph, int[] visited, List<int> ans) {
        if (visited[curr] == 1) return true;
        if (visited[curr] == 2) return false;
        
        visited[curr] = 1;
        foreach (int next in graph[curr])
            if (dfs(next, graph, visited, ans)) return true;
        
        visited[curr] = 2;
        ans.Add(curr);
        
        return false;
    }    
}
// 815. Bus Routes
/*BFS Solution
Explanation:
The first part loop on routes and record stop to routes mapping in to_route.
The second part is general bfs. Take a stop from queue and find all connected route.
The hashset seen record all visited stops and we won't check a stop for twice.
We can also use a hashset to record all visited routes, or just clear a route after visit.
*/
public class Solution {
    public int NumBusesToDestination(int[][] routes, int source, int target) {
        int n = routes.Length;
        Dictionary<int, List<int>> to_routes = new Dictionary<int, List<int>>();
        for (int i = 0; i < routes.Length; ++i) {
            foreach (int j in routes[i]) {
                if (!to_routes.ContainsKey(j))
                    to_routes[j]= new List<int>();
                to_routes[j].Add(i);
            }
        }
        Queue<int[]> bfs = new Queue<int[]>();
        bfs.Enqueue(new int[] {source, 0});
        List<int> seen = new List<int>();
        seen.Add(source);
        bool[] seen_routes = new bool[n];
        while (bfs.Count != 0) {
            int stop = bfs.Peek()[0], bus = bfs.Peek()[1];
            bfs.Dequeue();
            if (stop == target) return bus;
            foreach (int i in to_routes[stop]) {
                if (seen_routes[i]) continue;
                foreach (int j in routes[i]) {
                    if (!seen.Contains(j)) {
                        seen.Add(j);
                        bfs.Enqueue(new int[] {j, bus + 1});
                    }
                }
                seen_routes[i] = true;
            }
        }
        return -1;
    }
}
/*
Solution: BFS
Time Complexity: O(m*n) m: # of buses, n: # of routes
Space complexity: O(m*n + m)
*/
public class Solution {
    public int NumBusesToDestination(int[][] routes, int S, int T) {
        if (S == T) return 0;
    
    Dictionary<int, List<int>> m = new Dictionary<int, List<int>>();
    for (int i = 0; i < routes.Length; ++i)
      foreach (int stop in routes[i]){
          if (!m.ContainsKey(stop)) m.Add(stop,new List<int>());
           m[stop].Add(i);
      }
       
    
    int[] visited = new int[routes.Length]; Array.Fill(visited, 0);
    Queue<int> q = new Queue<int>();
    q.Enqueue(S);
    int buses = 0;
    
    while (q.Count != 0) {
      int size = q.Count;      
      ++buses;
      while (size-- > 0 ) {
        int curr = q.Peek(); q.Dequeue();        
        foreach (int bus in m[curr]) {
          if (visited[bus] > 0) continue;          
          visited[bus] = 1;
          foreach (int stop in routes[bus]) {
            if (stop == T) return buses;            
            q.Enqueue(stop);
          }
        }        
      }      
    }
    return -1;
    }
}
// Day 12 Dynamic Programming
// 198. House Robber

// 322. Coin Change

// Day 13 Dynamic Programming
// 416. Partition Equal Subset Sum
public class Solution {
    public bool CanPartition(int[] nums) {
        int sum = nums.Sum();
        if (sum % 2 != 0) return false;
        int[] dp = new int[sum + 1];Array.Fill(dp, 0);
        dp[0] = 1;
        foreach (int num in nums) {
            for (int i = sum; i >= 0; --i)
                if (dp[i] > 0) dp[i + num] = 1;
            if (dp[sum / 2] > 0) return true;
        }
        return false;
    }
}
// 152. Maximum Product Subarray

// Day 14 Sliding Window/Two Pointer
// 3. Longest Substring Without Repeating Characters

// 16. 3Sum Closest
//https://leetcode.com/problems/minimum-window-substring/discuss/26808/Here-is-a-10-line-template-that-can-solve-most-'substring'-problems
/*
Solution: Sorting + Two Pointers
Similar to LeetCode 15. 3Sum

Time complexity: O(n^2)
Space complexity: O(1)
*/

// 76. Minimum Window Substring
// based on templated code with optimisation (using array instead of map). The runtime dropped for e.g. in min window from 143ms -> 7ms.
// Minimum window
public class Solution {
    public string MinWindow(string s, string t) {
        int [] map = new int[128];
    foreach (char c in t.ToCharArray()) {
      map[c]++;
    }
    int start = 0, end = 0, minStart = 0, minLen = int.MaxValue, counter = t.Length;
    while (end < s.Length) {
      char c1 = s[end];
      if (map[c1] > 0) counter--;
      map[c1]--;
      end++;
      while (counter == 0) {
        if (minLen > end - start) {
          minLen = end - start;
          minStart = start;
        }
        char c2 = s[start];
        map[c2]++;
        if (map[c2] > 0) counter++;
        start++;
      }
    }

    return minLen == int.MaxValue ? "" : s.Substring(minStart, minLen);
    }
}
//using two pointers + HashMap <--very slow
public class Solution {
    public string MinWindow(string s, string t) {
        if(s == null || s.Length < t.Length || s.Length == 0){
        return "";
    }
    Dictionary<char,int> map = new Dictionary<char,int>();
    foreach(char c in t.ToCharArray()){
        if(map.ContainsKey(c)){
            map[c] = map[c]+1;
        }else{
            map[c] = 1;
        }
    }
    int left = 0;
    int minLeft = 0;
    int minLen = s.Length+1;
    int count = 0;
    for(int right = 0; right < s.Length; right++){
        if(map.ContainsKey(s[right])){
            map[s[right]] = map[s[right]]-1;
            if( map[s[right]] >= 0){
                count ++;
            }
            while(count == t.Length){
                if(right-left+1 < minLen){
                    minLeft = left;
                    minLen = right-left+1;
                }
                if(map.ContainsKey(s[left])){
                    map[s[left]] = map[s[left]] +1;
                    if(map[s[left]] > 0){
                        count --;
                    }
                }
                left ++ ;
            }
        }
    }
    if(minLen>s.Length)  
    {  
        return "";  
    }  
    
    return s.Substring(minLeft,minLen);
    }
}
// Day 15 Tree
// 100. Same Tree
/*
Solution: Recursion
Time complexity: O(n)
Space complexity: O(n)*/
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int val=0, TreeNode left=null, TreeNode right=null) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
public class Solution {
    public bool IsSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        return p.val == q.val && IsSameTree(p.left, q.left) && IsSameTree(p.right, q.right);
    }
}

public class Solution {
    public bool IsSameTree(TreeNode p, TreeNode q) {
        // p and q are both null
        if (p == null && q == null) return true;
        // one of p and q is null
        if (q == null || p == null) return false;
        if (p.val != q.val) return false;
        return IsSameTree(p.right, q.right) &&
                IsSameTree(p.left, q.left);
    }
}
// 101. Symmetric Tree

// 199. Binary Tree Right Side View

// Day 16 Design
// 232. Implement Queue using Stacks

// 155. Min Stack

// 208. Implement Trie (Prefix Tree)
public class Trie {

    public class TrieNode {
        public TrieNode() {
            children = new TrieNode[26];
            is_word = false;
        }
        public bool is_word;
        public TrieNode[] children;
    }
    
    private TrieNode root;
    public Trie() {
     /** Initialize your data structure here. */
        root = new TrieNode();
    }
    
    public void Insert(string word) {
         /** Inserts a word into the trie. */
        TrieNode p = root;
        for (int i = 0; i < word.Length; i++) {
            int index = (int)(word[i] - 'a');
            if (p.children[index] == null)
                p.children[index] = new TrieNode();
            p = p.children[index];
        }
        p.is_word = true;
    }
    
    public bool Search(string word) {
        /** Returns if the word is in the trie. */
        TrieNode node = find(word);
        return node != null && node.is_word;
    }
    
    public bool StartsWith(string prefix) {
         /** Returns if there is any word in the trie that starts with the given prefix. */
        TrieNode node = find(prefix);
        return node != null;
    }
     public TrieNode find(string prefix) {
        TrieNode p = root;
        for(int i = 0; i < prefix.Length; i++) {
            int index = (int)(prefix[i] - 'a');
            if (p.children[index] == null) return null;
            p = p.children[index];
        }
        return p;
    }
}
/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.Insert(word);
 * bool param_2 = obj.Search(word);
 * bool param_3 = obj.StartsWith(prefix);
 */
// Day 17 Interval
// 57. Insert Interval
public class Solution {
    public int[][] Insert(int[][] intervals, int[] newInterval) {
         
         List<int[]> l = new List<int[]>();
         List<int[]> r = new List<int[]>();
        int start = newInterval[0];
        int end = newInterval[1];
        foreach (int[] interval in intervals) {
            if (interval[1] < start)
                l.Add(interval);
            else if (interval[0] > end)
                r.Add(interval);
            else {
                start = Math.Min(start, interval[0]);
                end = Math.Max(end, interval[1]);
            }                
        }
       
       List<int[]> ans = l;
        ans.Add(new int[]{start,end});
        //ans.AddRange(r);
        //return ans.ToArray();
        return ans.Concat(r).ToArray(); //works too!
    }
}

public class Solution {
    public int[][] Insert(int[][] intervals, int[] newInterval) {
         List<int[]> result = new List<int[]>();
        
         foreach(int[] i in intervals){
             if(newInterval == null || i[1] < newInterval[0]){
                 result.Add(i);
             }else if(i[0] > newInterval[1]){
                // be carefult the sequence here
                 result.Add(newInterval);
                 result.Add(i);
                 newInterval = null;
             }else{
                 
                 newInterval[0] = Math.Min(newInterval[0], i[0]);//get min
                 newInterval[1] = Math.Max(newInterval[1], i[1]);//get max
             }
         }
        
        if(newInterval != null)
            result.Add(newInterval);
        
        return result.ToArray();
    }
}

// 56. Merge Intervals

// Day 18 Stack
// 735. Asteroid Collision
/*
Simulation
Time complexity: O(n), Space complexity: O(n)*/
public class Solution {
    public int[] AsteroidCollision(int[] a) {
        List<int> s = new List<int>();
        for (int i = 0 ; i < a.Length; ++i) {
            int size = a[i];
            if (size > 0) { // To right, OK
                s.Add(size);
            } else {
                // To left
                if (s.Count == 0 || s[s.Count-1] < 0) // OK when all are negtives
                    s.Add(size);
                else if (Math.Abs( s[s.Count-1]) <=Math.Abs(size)) {
                    // Top of the stack is going right.
                    // Its size is less than the current one.
                    
                    // The current one still alive!
                    if (Math.Abs( s[s.Count-1]) < Math.Abs(size)) --i;
                    
                    s.RemoveAt(s.Count-1); // Destory the top one moving right
                }                    
            }
        }
        
        // s must look like [-s1, -s2, ... , si, sj, ...]
        return s.ToArray();
    }
}
// Stack
// Time complexity: O(n), space complexity: O(n). n is number of asteroids.
public class Solution {
    public int[] AsteroidCollision(int[] a) {
        Stack<int> stack = new Stack<int>();
        for (int i = 0; i < a.Length; i++) {
            if (stack.Count == 0 || a[i] > 0) {
                stack.Push(a[i]);
                continue;
            }
            
            while (true) {
                int prev = stack.Peek();
                if (prev < 0) {
                    stack.Push(a[i]);
                    break;
                }
                if (prev == -a[i]) {
                    stack.Pop();
                    break;
                }
                if (prev > -a[i]) {
                    break;
                }
                stack.Pop();
                if (stack.Count == 0) {
                    stack.Push(a[i]);
                    break;
                }
            }
        }
        
        int[] res = new int[stack.Count];
        for (int i = stack.Count - 1; i >= 0; i--) {
            res[i] = stack.Pop();
        }
        
        return res;
    }
}

// 227. Basic Calculator II
/*Approach 2: Optimised Approach without the stack
Complexity Analysis

Time Complexity: O(n), 
where nn is the length of the string ss.

Space Complexity: O(1), 
as we use constant extra space to store lastNumber, result and so on.
*/
public class Solution {
    public int Calculate(string s) {
         if (s == null || s.Length == 0) return 0;
        int length = s.Length;
        int currentNumber = 0, lastNumber = 0, result = 0;
        char operation = '+';
        for (int i = 0; i < length; i++) {
            char currentChar = s[i];
            if (char.IsDigit(currentChar)) {
                currentNumber = (currentNumber * 10) + (currentChar - '0');
            }
            if (!char.IsDigit(currentChar) && currentChar != ' ' || i == length - 1) {
                if (operation == '+' || operation == '-') {
                    result += lastNumber;
                    lastNumber = (operation == '+') ? currentNumber : -currentNumber;
                } else if (operation == '*') {
                    lastNumber = lastNumber * currentNumber;
                } else if (operation == '/') {
                    lastNumber = lastNumber / currentNumber;
                }
                operation = currentChar;
                currentNumber = 0;
            }
        }
        result += lastNumber;
        return result;
    }
}

public class Solution {
    public int Calculate(string s) {
        int res = 0, num = 0, tmp = 0;
        char opr = '+';
        foreach (char chr in (s + "+").ToCharArray()) {
            if(chr == ' ')continue;
            if(char.IsDigit(chr)) {
                num = num * 10 + (chr - '0');
            } else {        
                switch(opr) {
                    case '+':
                        res += tmp;
                        tmp = num;
                        break;
                    case '-':
                        res += tmp;
                        tmp = -num;
                        break;
                    /* THIS WORKS TOO!
                    case '-':
                    case '+':
                        res += tmp;
                        tmp = num*(opr == '+' ? 1 : -1);
                        break;
                    */
                    case '*':
                        tmp *= num;
                        break;
                    case '/':
                        tmp /= num;
                        break;
                    default:
                        return -1;
                }
                num = 0;
                opr = chr;
            }
        }
        res += tmp;
        
        return res;
    }
}
/*
Solution: Stack
if operator is ‘+’ or ‘-’, push the current num * sign onto stack.
if operator ‘*’ or ‘/’, pop the last num from stack and * or / by the current num and push it back to stack.

The answer is the sum of numbers on stack.

3+2*2 => {3}, {3,2}, {3, 2*2} = {3, 4} => ans = 7
3 +5/2 => {3}, {3,5}, {3, 5/2} = {3, 2} => ans = 5
1 + 2*3 – 5 => {1}, {1,2}, {1,2*3} = {1,6}, {1, 6, -5} => ans = 2

Time complexity: O(n)
Space complexity: O(n)
*/
public class Solution {
    public int Calculate(string s) {
         List<int> nums = new List<int>();    
        char op = '+';
        int cur = 0;
        int pos = 0;
        while (pos < s.Length) {
          if (s[pos] == ' ') {
            pos+=1;
            continue;
          }
          while( pos < s.Length && char.IsDigit(s[pos])){
            // in the while loop and condition 
            // It's import to check pos < s.Length first then 
            // check the char.IsDigit
              cur = cur * 10 + (s[pos] - '0'); 
              pos+=1;
          }

          if (op == '+' || op == '-') {
            nums.Add(cur * (op == '+' ? 1 : -1));
          } else if (op == '*') {
            nums[nums.Count-1] *= cur;
          } else if (op == '/') {
            nums[nums.Count-1] /= cur;
          }

          cur = 0;      

          if(pos < s.Length){
               op = s[pos];  
            }
           pos+=1;
        }
        return nums.Sum();
    }
}
//Stack
public class Solution {
    public int Calculate(string s) {
       var stack = new Stack<int>();
        
        char op = '+';
        int num = 0;
        
        for (int i = 0; i <= s.Length; i++)
        {
            char c = i < s.Length ? s[i] : '+';
            
            if (c == ' ') continue;
            
            if (char.IsNumber(c))
            {
                num = num * 10 + (c - '0');
            }
            else
            {
                if (op == '+') stack.Push(num);
                if (op == '-') stack.Push(-num);
                if (op == '*') stack.Push(stack.Pop() * num);
                if (op == '/') stack.Push(stack.Pop() / num);
                
                op = c;
                
                num = 0;
            }
        }
        
        return stack.Sum();
    }
}
//List
public class Solution {
    public int Calculate(string s) {
        var stack = new List<int>();
        
        char op = '+';
        int num = 0;
        //it is to push the last number on the stack. Otherwise you would need to do code outside of the for. 
        //(which would be most of the code included in the else block)
        for (int i = 0; i <= s.Length; i++)
        {
            char c = i < s.Length ? s[i] : '+';
            
            if (c == ' ') continue;
            
            if (char.IsDigit(c))
            {
                num = num * 10 + (c - '0');
            }
            else
            {
                if (op == '+') stack.Add(num);
                if (op == '-') stack.Add(-num);
                if (op == '*') stack[stack.Count -1] *= num;
                if (op == '/') stack[stack.Count -1] /= num;
                
                op = c;
                
                num = 0;
            }
        }
        
        return stack.Sum();
    }
}
//Linq --> very slow
using System.Text.RegularExpressions;
public class Solution {
    public int Calculate(string s)=> s.Contains('+') ? s.Split('+').Select(Calculate).Sum()
            : (
                s.Contains('-') ? s.Split('-').Select((s, i) => Calculate(s) * (i == 0 ? 1 : -1)).Sum()
                : Regex.Matches('*' + s, @"([/*])(\s*)(\d*)")
                        .Select(m => (op: m.Groups[1].Value, number: Int32.Parse(m.Groups[3].Value)))
                        .Aggregate(1, (res, cur) => cur.op == "*" ? res * cur.number : res / cur.number)
              );
}
// Day 19 Union Find
// 547. Number of Provinces

// 947. Most Stones Removed with Same Row or Column
/* Count the Number of Islands, O(N)
I said it's a hard problem, LC rated it as medium.

Problem:
we can remove a stone if and only if,
there is another stone in the same column OR row.
We try to remove as many as stones as possible.


One sentence to solve:
Connected stones can be reduced to 1 stone,
the maximum stones can be removed = stones number - islands number.
so just count the number of "islands".


1. Connected stones
Two stones are connected if they are in the same row or same col.
Connected stones will build a connected graph.
It's obvious that in one connected graph,
we can't remove all stones.

We have to have one stone left.
An intuition is that, in the best strategy, we can remove until 1 stone.

I guess you may reach this step when solving the problem.
But the important question is, how?


2. A failed strategy
Try to remove the least degree stone
Like a tree, we try to remove leaves first.
Some new leaf generated.
We continue this process until the root node left.

However, there can be no leaf.
When you try to remove the least in-degree stone,
it won't work on this "8" like graph:
[[1, 1, 0, 0, 0],
[1, 1, 0, 0, 0],
[0, 1, 1, 0, 0],
[0, 0, 1, 1, 1],
[0, 0, 0, 1, 1]]

The stone in the center has least degree = 2.
But if you remove this stone first,
the whole connected stones split into 2 parts,
and you will finish with 2 stones left.


3. A good strategy
In fact, the proof is really straightforward.
You probably apply a DFS, from one stone to next connected stone.
You can remove stones in reversed order.
In this way, all stones can be removed but the stone that you start your DFS.

One more step of explanation:
In the view of DFS, a graph is explored in the structure of a tree.
As we discussed previously,
a tree can be removed in topological order,
from leaves to root.


4. Count the number of islands
We call a connected graph as an island.
One island must have at least one stone left.
The maximum stones can be removed = stones number - islands number

The whole problem is transferred to:
What is the number of islands?

You can show all your skills on a DFS implementation,
and solve this problem as a normal one.


5. Unify index
Struggle between rows and cols?
You may duplicate your codes when you try to the same thing on rows and cols.
In fact, no logical difference between col index and rows index.

An easy trick is that, add 10000 to col index.
So we use 0 ~ 9999 for row index and 10000 ~ 19999 for col.


6. Search on the index, not the points
When we search on points,
we alternately change our view on a row and on a col.

We think:
a row index, connect two stones on this row
a col index, connect two stones on this col.

In another view：
A stone, connect a row index and col.

Have this idea in mind, the solution can be much simpler.
The number of islands of points,
is the same as the number of islands of indexes.


7. Union-Find
I use union find to solve this problem.
As I mentioned, the elements are not the points, but the indexes.

for each point, union two indexes.
return points number - union number
Copy a template of union-find,
write 2 lines above,
you can solve this problem in several minutes.


Complexity
union and find functions have worst case O(N), amortize O(1)
The whole union-find solution with path compression,
has O(N) Time, O(N) Space

If you have any doubts on time complexity,
please refer to wikipedia first.
*/
public class Solution {
    Dictionary<int, int> f = new Dictionary<int,int>();
    int islands = 0;
    public int RemoveStones(int[][] stones) {
        for (int i = 0; i < stones.Length; ++i)
            union(stones[i][0], ~stones[i][1]);
        return stones.Length - islands;
    }
    
    public int find(int x) {
        if (!f.ContainsKey(x) ){
            f[x] = x;
            islands++;
        }
            
        if (x != f[x])
            f[x] = find(f[x]);
        return f[x];
    }
    public void union(int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y) {
            f[x] = y;
            islands--;
        }
    }
}

/*
Update About Union Find Complexity
I have 3 main reasons that always insist O(N), on all my union find solutions.

The most important, union find is really a common knowledge for algorithm.
Using both path compression, splitting, or halving and union by rank or size ensures
that the amortized time per operation is only O(1).
So it's fair enough to apply this conclusion.

It's really not my job to discuss how union find works or the definition of big O.
I bet everyone can find better resource than my post on this part.
You can see the core of my solution is to transform the problem as a union find problem.
The essence is the thinking process behind.
People can have their own template and solve this problem with 2-3 more lines.
But not all the people get the point.

I personally manually write this version of union find every time.
It is really not worth a long template.
The version with path compression can well handle all cases on leetcode.
What‘s the benefit here to add more lines?

In this problem, there is N union operation, at most 2 * sqrt(N) node.
When N get bigger, the most operation of union operation is amortize O(1).

I knew there were three good resourse of union find:

top down analusis of path compression
wiki
stackexchange
But they most likely give a upper bound time complexity of union find,
not a supreme.
If anyone has a clear example of union find operation sequence,
to make it larger than O(N), I am so glad to know it.
*/
// Day 20 Brute Force/Backtracking
// 39. Combination Sum

// 46. Permutations
