// Day 1 Binary Search
//35. Search Insert Position
//Find the number or upper bound if doesn’t exist.
//Time complexity: O(logn)
//Space complexity: O(1)
public class Solution {
    public int SearchInsert(int[] nums, int target) {
        int l = 0;
        int r = nums.Length;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (nums[m] == target)
            {
                return m;
            }
            else if(nums[m] > target)
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return l;
    }
}

// 704. Binary Search
// Solution: Binary Search
// Time complexity: O(logn)
// Space complexity: O(1)
public class Solution {
    public int Search(int[] nums, int target) {
        int l = 0;
        int r = nums.Length;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (nums[m] == target)
            {
                return m;
            }
            else if(nums[m] > target)
            {
                r = m;
            }
            else
            {
                l = m+1;
            }
        }
        return -1;
    }
}

// 278. First Bad Version
/* The isBadVersion API is defined in the parent class VersionControl.
      bool IsBadVersion(int version); */

public class Solution : VersionControl {
    public int FirstBadVersion(int n) {
        int l = 0;
        int r = n;
        while (l<r)
        {
            int m = l+(r-l)/2;
            if (IsBadVersion(m) == true)
            {
                r = m;
            }
            else if(IsBadVersion(m) == false)
            {
                l = m+1;
            }

        }
        return l;
    }
}

// Day 2 Two Pointers
// 189. Rotate Array
//Could you do it in-place with O(1) extra space
public class Solution {
    public void Rotate(int[] nums, int k) {
        if (k > nums.Length){
            k = k % nums.Length;
        }
        
        if (k >0 )
        {
            reverse(nums, 0, nums.Length-1);
            reverse(nums, 0, k-1);
            reverse(nums, k, nums.Length-1);
        }
    }
    public void reverse(int[] nums, int i, int j)
    {
        int tmp = 0;
        while(i<j)
        {
            tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
}

// 977. Squares of a Sorted Array
public class Solution {
    public int[] SortedSquares(int[] nums) {
        int[] answer = new int[nums.Length];
        int l = 0;
        int r = nums.Length - 1;
        while (l<=r)
        {
            if (Math.Abs(nums[l]) < Math.Abs(nums[r]))
            {
                answer[r-l] = nums[r]*nums[r];
                r--;
            }
            else{
                answer[r-l] = nums[l]*nums[l];
                l++;
            }
        }
        return answer;
    }
}

// Day 3 Two Pointers
// 283. Move Zeroes
// [0,1,0,3,12] ->[0,1,0,3,12]->[1,0,0,3,12]->[1,3,0,0,12]->[1,3,12,0,0]
public class Solution {
    public void MoveZeroes(int[] nums) {
        int count = 0; 
        for (int i=0;i<nums.Length;i++)
        {
	        if (nums[i]==0)
            {
                count++; 
            }
            else if (count > 0) 
            {
	            int t = nums[i];
	            nums[i]=0;
	            nums[i-count]=t;
            }
        }
    }
}

// 167. Two Sum II - Input Array Is Sorted 
public class Solution {
    public int[] TwoSum(int[] numbers, int target) {
        int l = 0,  r = numbers.Length-1;
        while(l<r){
            if(numbers[l]+numbers[r]==target)
            {
                return new int[]{l+1,r+1};
            }
            else if(numbers[l]+numbers[r]>target){
                r--;
            }
            else
            {
                l++;
            }
        }
        return new int[]{};
    }
}

// Day 4 Two Pointers
// 344. Reverse String
public class Solution {
    public void ReverseString(char[] s) {
        int i = 0, j = s.Length -1;
        while(i<j)
        {
            char temp = s[i];
            s[i] = s[j];
            s[j] = temp;
            i++;
            j--;
        }
    }
}

// 557. Reverse Words in a String III
public class Solution {
    public string ReverseWords(string s) {
        String[] strs = s.Split(" "); //splict the string
        Array.Reverse( strs ); // reverse the string list
        //string sr = string.Join(" ", strs); //join together the string list as string
        //return new string(sr.ToCharArray().Reverse().ToArray());// reverse the whole string
        return new string(string.Join(" ", strs).ToCharArray().Reverse().ToArray());
    }
}

public class Solution {
    public string ReverseWords(string s) {
      String[] strs = s.Split(" ");
      StringBuilder sb = new StringBuilder();
      String space = "";
      for(int i = 0; i<strs.Length;i++){
        sb.Append(space);
        sb.Append(reverse(strs[i]));  
        space = " ";
      }
      return String.Concat(sb); //String.Join(sb)
    }
    
    public String reverse(String str){
      char[] sc = str.ToCharArray();
      int s = 0, e = sc.Length - 1;
      while(s < e){
        char temp = sc[s];
        sc[s] = sc[e];
        sc[e] = temp;
        s++;
        e--;
      }
      return new String(sc);
    }
}
// Day 5 Two Pointers
// 876. Middle of the Linked List
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
//Each time, slow go 1 steps while fast go 2 steps.
//When fast arrives at the end, slow will arrive right in the middle.
/*
    if len = odd :

    head -> 1 -> 2 -> 3 -> 4 -> 5 -> Null

    fast = 1,3,5 -----> for this we have to use stopping case fast != Null
    slow = 1,2,3

    if len = even :

    head -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> Null

    fast = 1,3,5,Null -----> for this we have to use stopping case fast.next != Null
    slow = 1,2,3,4
*/
public class Solution {
    public ListNode MiddleNode(ListNode head) {
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        return slow;
    }
}

// 19. Remove Nth Node From End of List
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
    public ListNode RemoveNthFromEnd(ListNode head, int n) {
        ListNode fast = head, slow = head;
        for (int i = 0; i < n; i++) fast = fast.next;
        if (fast == null) return head.next;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }
}

// Day 6 Sliding Window
// 3. Longest Substring Without Repeating Characters

public class Solution {
    public int LengthOfLongestSubstring(string s) {
        Dictionary<Char, int> map = new Dictionary<Char, int>();
        int max = 0, start = -1;
        for (int i=0; i<s.Length; i++){
            if (map.ContainsKey(s[i])){
                start = Math.Max(start,map[s[i]]);
            }
            map[s[i]]=i;
            max = Math.Max(max,i-start);
        }
        return max;

    }
}

// 567. Permutation in String
/*
How do we know string p is a permutation of string s? 
Easy, each character in p is in s too. 
So we can abstract all permutation strings of s to a map (Character -> Count). i.e. abba -> {a:2, b:2}. 
Since there are only 26 lower case letters in this problem, 
we can just use an array to represent the map.

How do we know string s2 contains a permutation of s1? 
We just need to create a sliding window with length of s1, 
move from beginning to the end of s2. 
When a character moves in from right of the window, 
we subtract 1 to that character count from the map. 
When a character moves out from left of the window, 
we add 1 to that character count. 
So once we see all zeros in the map, meaning equal numbers of every characters 
between s1 and the substring in the sliding window, we know the answer is true.
*/
public class Solution {
    public bool CheckInclusion(string s1, string s2) {
        int len1 = s1.Length, len2 = s2.Length;
        if (len1 > len2) return false;

        int[] count = new int[26];
        for (int i = 0; i < len1; i++) {
            count[s1[i] - 'a']++;
        }

        for (int i = 0; i < len2; i++) {
            count[s2[i] - 'a']--;
            if(i - len1 >= 0) count[s2[i - len1] - 'a']++;
            if (allZero(count)) return true;
        }

        return false;
    }

    private bool allZero(int[] count) {
        for (int i = 0; i < 26; i++) {
            if (count[i] != 0) return false;
        }
        return true;
    }
}

/*
In first loop we assume that s1 and s2 has same length, 
we use a map, for every char of s1 we add to map, 
for every char of s2 we delete from map. 
After that we check if for every char in map, 
we have a perfect balance.(each char has count zero)

In second loop we start to move the windows from left to right. 
Each step we deal with the head and tail of the window, 
then check if the map has a balance.
*/
public class Solution {
    public bool CheckInclusion(string s1, string s2) {
      int l1 = s1.Length, l2 = s2.Length;
      if(l1 > l2) return false;

      int[] map = new int[26];
      char[] s1c = s1.ToCharArray(), s2c = s2.ToCharArray();
      for(int i = 0; i < l1; i++){
        map[s1c[i] - 'a']++;
        map[s2c[i] - 'a']--;
      }
      
      if(isAllZero(map)) return true;
      
      for(int i = 0; i < l2 - l1; i++){
        map[s2c[i] - 'a']++;
        map[s2c[i + l1] - 'a']--;
        if(isAllZero(map)) return true;
      }
      
      return false;
    }
    
    public bool isAllZero(int[] array)
    {
      foreach(int a in array){
        if(a > 0) return false;  
      }    
      return true;
    }
}
/*
Approach 6 Optimized Sliding Window [Accepted]:
Algorithm
The last approach can be optimized, 
if instead of comparing all the elements of the hashmaps for every updated s2map
corresponding to every window of s2s2 considered, 
we keep a track of the number of elements which were already matching in the earlier hashmap 
and update just the count of matching elements when we shift the window towards the right.

To do so, we maintain a countcount variable, which stores the number of characters(out of the 26 alphabets), 
which have the same frequency of occurence in s1s1 and the current window in s2s2. 
When we slide the window, if the deduction of the last element 
and the addition of the new element leads to a new frequency match of any of the characters, 
we increment the countcount by 1. 
If not, we keep the countcount intact. 
But, if a character whose frequency was the same earlier(prior to addition and removal) is added,
it now leads to a frequency mismatch which is taken into account by 
decrementing the same countcount variable. 
If, after the shifting of the window, the countcount evaluates to 26, 
it means all the characters match in frequency totally. 
So, we return a True in that case immediately.

Complexity Analysis

Time complexity: O(l1+(l2−l1)). 
Where l1 is the length of string s1 and l2 is the length of string s2.
Space complexity: O(1). Constant space is used.

*/
public class Solution {
    public bool CheckInclusion(string s1, string s2) {
        if (s1.Length > s2.Length) return false;
        int[] s1map = new int[26];
        int[] s2map = new int[26];
        for (int i = 0; i < s1.Length; i++) {
            s1map[s1[i] - 'a']++;
            s2map[s2[i] - 'a']++;
        }
        
        int count = 0;
        for (int i = 0; i < 26; i++)
            if (s1map[i] == s2map[i])
                count++;
                
        for (int i = 0; i < s2.Length - s1.Length; i++) {
            int r = s2[i + s1.Length] - 'a', l = s2[i] - 'a';
            if (count == 26)
                return true;
            s2map[r]++;
            if (s2map[r] == s1map[r])
                count++;
            else if (s2map[r] == s1map[r] + 1)
                count--;
            s2map[l]--;
            if (s2map[l] == s1map[l])
                count++;
            else if (s2map[l] == s1map[l] - 1)
                count--;
        }
        return count == 26;
    }
}

// Day 7 Breadth-First Search / Depth-First Search
// 733. Flood Fill
// Time complexity: O(m*n), space complexity: O(1). m is number of rows, n is number of columns.
public class Solution {
    public int[][] FloodFill(int[][] image, int sr, int sc, int newColor) {
        if (image[sr][sc] == newColor) return image;
        fill(image, sr, sc, image[sr][sc], newColor);
        return image;
    }
    public void fill(int[][] image, int sr, int sc, int color, int newColor){
        if (sr < 0 || sr >= image.Length || sc < 0 || sc >= image[0].Length || image[sr][sc] != color) return;
        image[sr][sc] = newColor;
        fill(image, sr + 1, sc, color, newColor);
        fill(image, sr - 1, sc, color, newColor);
        fill(image, sr, sc + 1, color, newColor);
        fill(image, sr, sc - 1, color, newColor);
        
    }
}

// Approach 1: Depth-First Search
public class Solution {
    public int[][] FloodFill(int[][] image, int sr, int sc, int newColor) {
        int color = image[sr][sc];
        if (color != newColor) dfs(image, sr, sc, color, newColor);
        return image;
    }
    public void dfs(int[][] image, int r, int c, int color, int newColor) {
        if (image[r][c] == color) {
            image[r][c] = newColor;
            if (r >= 1) dfs(image, r-1, c, color, newColor);
            if (c >= 1) dfs(image, r, c-1, color, newColor);
            if (r+1 < image.Length) dfs(image, r+1, c, color, newColor);
            if (c+1 < image[0].Length) dfs(image, r, c+1, color, newColor);
        }
    }
}

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
        if( i >= 0 && i < grid.Length && j >= 0 && j < grid[0].Length && grid[i][j] == 1){
            grid[i][j] = 0;
            return 1 + AreaOfIsland(grid, i+1, j) + AreaOfIsland(grid, i-1, j) + AreaOfIsland(grid, i, j-1) + AreaOfIsland(grid, i, j+1);
        }
        return 0;
    }
}
/*
Complexity Analysis
Time Complexity: O(R*C), where RR is the number of rows in the given grid,
 and CC is the number of columns. We visit every square once.
Space complexity: O(R∗C), the space used by seen to keep track of visited squares,
 and the space used by the call stack during our recursion.
*/
public class Solution {
    int[][] grid;
    bool[,] seen;

    public int area(int r, int c) {
        if (r < 0 || r >= grid.Length || c < 0 || c >= grid[0].Length ||
                seen[r,c] || grid[r][c] == 0)
            return 0;
        seen[r,c] = true;
        return (1 + area(r+1, c) + area(r-1, c)
                  + area(r, c-1) + area(r, c+1));
    }

    public int MaxAreaOfIsland(int[][] grid) {
        this.grid = grid;
        seen = new bool[grid.Length, grid[0].Length];
        int ans = 0;
        for (int r = 0; r < grid.Length; r++) {
            for (int c = 0; c < grid[0].Length; c++) {
                ans = Math.Max(ans, area(r, c));
            }
        }
        return ans;
    }
}
/*
We can try the same approach using a stack based, (or "iterative") depth-first search.
Here, seen will represent squares that have either been visited or are added to 
our list of squares to visit (stack). For every starting land square that hasn't been visited, 
we will explore 4-directionally around it, adding land squares that haven't been added to seen to our stack.

On the side, we'll keep a count shape of the total number of squares seen during the exploration of this shape. 
We'll want the running max of these counts.
*/
public class Solution {
    public int MaxAreaOfIsland(int[][] grid) {
        bool[,] seen = new bool[grid.Length, grid[0].Length];
        int[] dr = new int[]{1, -1, 0, 0};
        int[] dc = new int[]{0, 0, 1, -1};

        int ans = 0;
        for (int r0 = 0; r0 < grid.Length; r0++) {
            for (int c0 = 0; c0 < grid[0].Length; c0++) {
                if (grid[r0][c0] == 1 && !seen[r0,c0]) {
                    int shape = 0;
                    Stack<int[]> stack = new Stack<int[]>();
                    stack.Push(new int[]{r0, c0});
                    seen[r0,c0] = true;
                    while (stack.Count != 0) {
                        int[] node = stack.Pop();
                        int r = node[0], c = node[1];
                        shape++;
                        for (int k = 0; k < 4; k++) {
                            int nr = r + dr[k];
                            int nc = c + dc[k];
                            if (0 <= nr && nr < grid.Length &&
                                    0 <= nc && nc < grid[0].Length &&
                                    grid[nr][nc] == 1 && !seen[nr,nc]) {
                                stack.Push(new int[]{nr, nc});
                                seen[nr,nc] = true;
                            }
                        }
                    }
                    ans = Math.Max(ans, shape);
                }
            }
        }
        return ans;
    }
}

// Day 8 Breadth-First Search / Depth-First Search
// 617. Merge Two Binary Trees

// Tree Traversal

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
    public TreeNode MergeTrees(TreeNode root1, TreeNode root2) {
        if(root1 == null && root2 == null) return null;
        else if(root1 == null) return root2;
        else if(root2 == null) return root1;
        TreeNode n= new TreeNode(root1.val+root2.val);
        n.left=MergeTrees(root1.left, root2.left);
        n.right=MergeTrees(root1.right, root2.right);
        return n;
    }
}


public class Solution {
    public TreeNode MergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null) return null;
        
        int val = (root1 == null ? 0 : root1.val) + (root2 == null ? 0 : root2.val);
        TreeNode newNode = new TreeNode(val);
        
        newNode.left = MergeTrees(root1 == null ? null : root1.left, root2 == null ? null : root2.left);
        newNode.right = MergeTrees(root1 == null ? null : root1.right, root2 == null ? null : root2.right);
    }
}
// 116. Populating Next Right Pointers in Each Node
/*
// Definition for a Node.
public class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}
*/

public class Solution {
    public Node Connect(Node root) {
        Node curr = root;
        while(curr != null && curr.left != null){// Escape loop through the last level with no left child and right child.
            Node nextLevelCurr = curr.left;
            while(curr != null){
                curr.left.next = curr.right;
                if(curr.next != null) curr.right.next = curr.next.left;
                else curr.right.next = null;
                curr = curr.next;
            }
            curr = nextLevelCurr;
        }
        return root;
    }
}

/*
// Definition for a Node.
public class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}
*/

public class Solution {
    public Node Connect(Node root) {
        if (root == null) {
            return null;}
        if (root.left != null) {
            root.left.next = root.right;
            if (root.next != null) {
                root.right.next = root.next.left;
            }
        }

        Connect(root.left);
        Connect(root.right);
        return root;
    }
    
}
// Day 9 Breadth-First Search / Depth-First Search
// 542. 01 Matrix
public class Solution {
    public int[][] UpdateMatrix(int[][] mat) {
        int m = mat.Length, n = mat[0].Length, INF = m + n; // The distance of cells is up to (M+N)
        for (int r = 0; r < m; r++) {
            for (int c = 0; c < n; c++) {
                if (mat[r][c] == 0) continue;
                int top = INF, left = INF;
                if (r - 1 >= 0) top = mat[r - 1][c];
                if (c - 1 >= 0) left = mat[r][c - 1];
                mat[r][c] = Math.Min(top, left) + 1;
            }
        }
        for (int r = m - 1; r >= 0; r--) {
            for (int c = n - 1; c >= 0; c--) {
                if (mat[r][c] == 0) continue;
                int bottom = INF, right = INF;
                if (r + 1 < m) bottom = mat[r + 1][c];
                if (c + 1 < n) right = mat[r][c + 1];
                mat[r][c] = Math.Min(mat[r][c], Math.Min(bottom, right) + 1);
            }
        }
        return mat;
    }
}

/*
Solution 1: DP
Two passes:
down, right
up, left
Time complexity: O(mn)
Space complexity: O(mn)
*/
public class Solution {
    public int[][] UpdateMatrix(int[][] mat) {
        int m = mat.Length,n = mat[0].Length;
        int[][] ans = new int[m][];
        for (int i = 0; i < m; i++){
            ans[i] = new int[n];
            for (int j = 0; j < n; j++){
                ans[i][j] = int.MaxValue-m*n;
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

// 994. Rotting Oranges
// 
public class Solution {
    public int OrangesRotting(int[][] grid) {
        //number of valid cells (cells with some orange)
        int ct = 0;
        //result
        int res = -1;
        //actually queue of pairs of coord i and j
        Queue<List<int>> q = new Queue<List<int>>();
         
        //ways to move
        List<List<int>> dir = new List<List<int>>(){new List<int>(){-1, 0}, new List<int>(){1, 0}, new List<int>(){0, -1}, new List<int>(){0, 1}};

        //create staring nodes to queue to do bfs
        for(int i = 0; i < grid.Length; i++) {
            for(int j = 0; j < grid[0].Length; j++) {
                //increasing number of valid cells
                if(grid[i][j] > 0) ct++;

                //only rotten oranges must be on initial step of queue
                if(grid[i][j] == 2) q.Enqueue(new List<int>(){i, j});
            }
        }

        //bfs
        while(q.Count != 0) {

            //we do one step from rotten
            res++;

            //see next comment
            int size = q.Count;

            //need to start from all rotten nodes at one moment 
            for(int k = 0; k < size; k++) {

                //take node from head
                List<int> cur = q.Peek();
                q.Dequeue();

                //number of valid decreasing
                ct--;

                //need to look through all four directions
                for(int i = 0; i < 4; i++) {
                    //taking coords
                    int x = cur[0] + dir[i][0];
                    int y = cur[1] + dir[i][1];

                    //if we go out of border or find non-fresh orange, we skip this iteration
                    if(x >= grid.Length || x < 0 || y >= grid[0].Length || y < 0 || grid[x][y] != 1) 
                        continue;

                    //orange becomes rotten
                    grid[x][y] = 2;

                    //this orange will make neighbor orange rotten
                    q.Enqueue(new List<int>(){x, y});
                }
            }
        }
        //if we looked through all oranges, return result, else -1
        return (ct == 0) ? Math.Max(0, res) : -1;
    }
}

// DFS - No extra space
/*
First check if the input is invalid.
Then, iterate over every cell in the matrix with rotAdjacent if the cell is rotten.

In rotAdjacent there has to be bounds checking:
1.) check if we are at the left or right edge
2.) check if we are at the top or bottom edge
3.) check if the cell is empty
4.) check if this cell has already been touched by another depth-first search using rotAdjacent 
that hit it faster than the original rot we are recursing from currently.
In these four cases we just end.

Otherwise, we store minutes, which represents the current time-step in the cell. 
Minutes at the first time step is 2 to offset that values 0 and 1 are reserved to 
indicate empty and fresh cells. Then we recursively invoke rotAdjacent 
on all adjacent cells, left, right, top, and bottom.

By starting recursive rotting from each rotten cell, we can see if it is possible to 
spread adjacently to all fresh cells. After we finish with all the rotAdjacent, 
we return to orangesRotting, and check if there's any fresh cells left, in which case we fail.

Otherwise, we get the largest value from the grid, 
which represents the timestep at which the final fresh cell was rotted, 
remove the offset of 2, and return.
*/
public class Solution {
    public int OrangesRotting(int[][] grid) {
        if(grid == null || grid.Length == 0) return -1;
        
        for(int i=0; i<grid.Length; i++) {
            for(int j=0; j<grid[0].Length; j++) {
                if(grid[i][j] == 2) rotAdjacent(grid, i, j, 2);
            }
        }
        
        int minutes = 2;
        foreach(int[] row in grid) {
            foreach(int cell in row) {
                if(cell == 1) return -1;
                minutes = Math.Max(minutes, cell);
            }
        }
        
        return minutes - 2;
    }
    
    private void rotAdjacent(int[][] grid, int i, int j, int minutes) {
        if(i < 0 || i >= grid.Length /* out of bounds */
          || j < 0 || j >= grid[0].Length /* out of bounds */
          || grid[i][j] == 0 /* empty cell */
          || (1 < grid[i][j] && grid[i][j] < minutes) /* this orange is already rotten by another rotten orange */
          ) return;
        else {
            grid[i][j] = minutes;
            rotAdjacent(grid, i - 1, j, minutes + 1);
            rotAdjacent(grid, i + 1, j, minutes + 1);
            rotAdjacent(grid, i, j - 1, minutes + 1);
            rotAdjacent(grid, i, j + 1, minutes + 1);
        }
    }
    
}

// Day 11 Recursion / Backtracking

// 77. Combinations
/*
Solution: DFS
Time complexity: O(C(n, k))
Space complexity: O(k)
*/
public class Solution {
    public IList<IList<int>> Combine(int n, int k) {
        IList<IList<int>> ans = new List<IList<int>>();
        List<int> cur = new List<int>();
        void dfs (int s) {
          if (cur.Count == k) {  // base case
            ans.Add(new List<int>(cur)); 
            return;
          }
          for (int i = s; i < n; ++i) {
            cur.Add(i + 1); //consider the current element i
            dfs(i + 1); // generate combinations
            cur.RemoveAt(cur.Count - 1); //proceed to next element
          }  
        };
        dfs(0);
        return ans; //return answer
    }
}

// 46. Permutations
// Solution: DFS
// Time complexity: O(n!)
// Space complexity: O(n)

public class Solution {
    public IList<IList<int>> Permute(int[] nums) {
        int n = nums.Length;
        IList<IList<int>> ans = new List<IList<int>>();
        List<int> used = new List<int>(new int[n]);
        List<int> path = new List<int>();
        void dfs (int d) {
          if (d == n) {
            ans.Add(new List<int>(path));
            return;
          }
          for (int i = 0; i < n; ++i) {
            if (used[i] > 0) continue;
            used[i] = 1;
            path.Add(nums[i]);
            dfs(d + 1);
            path.RemoveAt(path.Count-1);
            used[i] = 0;
          }
        };
        dfs(0);
        return ans;
    }
}

// 784. Letter Case Permutation

public class Solution {
    public IList<string> LetterCasePermutation(string s) {
    List<string> ans = new List<string>();
    StringBuilder sb = new StringBuilder(s);
    dfs(sb, 0, ans);
    return ans;
  }
    public void dfs(StringBuilder s, int i, List<string> ans) {
    if (i == s.Length) {
      ans.Add(s.ToString());
      return;      
    }
    dfs(s, i + 1, ans);    
    if (!Char.IsLetter(s[i])) return;
    s[i] = Convert.ToChar((int)s[i] ^ (1 << 5)); //s[i] ^= (1 << 5); (XOR) will always toggle the 5th bit which means 97 will become 65 and 65 will become 97;
    dfs(s, i + 1, ans);
    s[i] = Convert.ToChar((int)s[i] ^ (1 << 5));//p |= (1 << 5);
  }
}

// Day 12 Dynamic Programming

// 70. Climbing Stairs
public class Solution {
    public int ClimbStairs(int n) {
        int one = 1, two = 1, curr = 1;
        for(int i = 2; i<=n;i++)
        {
            curr = one + two;
            two = one;
            one = curr;
        }
        return curr;
        
    }
}

// 198. House Robber
public class Solution {
    public int Rob(int[] nums) {
        if (nums.Length == 0) return 0;
        int dp2 = 0;
        int dp1 = 0;
        for (int i = 0; i < nums.Length;i++) {
            int dp = Math.Max(dp2 + nums[i], dp1);
            dp2 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}

// 120. Triangle
public class Solution {
    public int MinimumTotal(IList<IList<int>> triangle) {
        /*
        // [[2]                 [[2]
        //  [3, 4]]              [5, 6]
        //  [6, 5, 7]]           [11, 10, 11]
        //  [4, 1, 8, 3]]        [15, 11, 18, 14]]
        */
        int n = triangle.Count;  
 
        // t[i][j] := minTotalOf(i,j)
        // t[i][j] += min(t[i - 1][j], t[i - 1][j - 1])

        for (int i = 0; i < n; ++i)
          for (int j = 0; j <= i; ++j) {
            if (i == 0 && j == 0) continue; //row 0
            if (j == 0) triangle[i][j] += triangle[i - 1][j]; // row 1
            else if (j == i) triangle[i][j] += triangle[i - 1][j - 1]; 
            else triangle[i][j] += Math.Min(triangle[i - 1][j], triangle[i - 1][j - 1]);
          }

        return triangle[triangle.Count-1].Min();
    }
}

// Day 13 Bit Manipulation
// 231. Power of Two
/*
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
*/
public class Solution {
    public bool IsPowerOfTwo(int n) {
         return n > 0 && (n & (n-1)) == 0;
    }
}
/*
Solution - (Recursive)
If a number is power of two, it can be recursively divided by 2 till it becomes 1
If the start number is 0 or if any intermediate number is not divisible by 2, we return false
Time Complexity : O(logn), where n is the given input number
Space Complexity : O(logn), required for recursive stack
*/
public class Solution {
    public bool IsPowerOfTwo(int n) {
        if(n == 0) return false;
        if(n == 1) return true;
        return n % 2 == 0 && IsPowerOfTwo(n / 2);
    }
}
/*
Solution - (Iterative)
The same solution as above but done iteratively
Time Complexity : O(logn), where n is the given input number
Space Complexity : O(1)
*/
public class Solution {
    public bool IsPowerOfTwo(int n) {
         if(n == 0) return false;
        while(n % 2 == 0) 
            n /= 2;
        return n == 1;
    }
}
/*
Solution - (Math)
Only a power of 2 will be able to divide a larger power of 2. Thus, we can take the largest power of 2 for our given range and check if n divides it
Time Complexity : O(1)
Space Complexity : O(1)
*/
public class Solution {
    public bool IsPowerOfTwo(int n) {
         return n > 0 && (1 << 31) % n == 0;
    }
}

// 191. Number of 1 Bits
public class Solution {
    public int HammingWeight(uint n) {
        int count = 0;
    
        while (n>0) {
            n &= (n - 1);
            count++;
        }

        return count;
    }
}
// Day 14 Bit Manipulation
// 190. Reverse Bits
/*
We first intitialize result to 0. We then iterate from
0 to 31 (an integer has 32 bits). In each iteration:
We first shift result to the left by 1 bit.
Then, if the last digit of input n is 1, we add 1 to result. To
find the last digit of n, we just do: (n & 1)
Example, if n=5 (101), n&1 = 101 & 001 = 001 = 1;
however, if n = 2 (10), n&1 = 10 & 01 = 00 = 0).

Finally, we update n by shifting it to the right by 1 (n >>= 1). 
This is because the last digit is already taken care of, 
so we need to drop it by shifting n to the right by 1.

At the end of the iteration, we return result.
*/
public class Solution {
    public uint reverseBits(uint n) {
        if (n == 0) return 0;
    
        uint result = 0;
        for (int i = 0; i < 32; i++) {
            result <<= 1;
            if ((n & 1) == 1) result++;
            n >>= 1;
        }
        return result;
    }
}
// 136. Single Number
public class Solution {
    public int SingleNumber(int[] nums) {
        int result = nums[0];
        for(int i=1; i<nums.Length; i++)
            result ^= nums[i]; /* Get the xor of all elements */
        return result;
    }
}

// Algorithm II
// Day 1 Binary Search
// 34. Find First and Last Position of Element in Sorted Array
 
// 33. Search in Rotated Sorted Array
/*
Explanation

Remember the array is sorted, except it might drop at one point.

If nums[0] <= nums[i], then nums[0..i] is sorted (in case of "==" it's just one element,
and in case of "<" there must be a drop elsewhere). So we should keep searching in nums[0..i] 
if the target lies in this sorted range, i.e., if nums[0] <= target <= nums[i].

If nums[i] < nums[0], then nums[0..i] contains a drop, 
and thus nums[i+1..end] is sorted and lies strictly between nums[i] and nums[0]. 
So we should keep searching in nums[0..i] if the target doesn't lie strictly between them, 
i.e., if target <= nums[i] < nums[0] or nums[i] < nums[0] <= target
*/
public class Solution {
    public int Search(int[] nums, int target) {
        int l = 0, r = nums.Length;
        while (l < r){
            int m = l+(r-l) / 2;
            if (nums[m] == target)
                return m;
            if ((nums[0] <= target && target < nums[m] )||( target < nums[m] && nums[m] < nums[0]) ||( nums[m] < nums[0]  &&  nums[0]<= target)){
                r = m;
            }
            else{
                l = m+1;
            } 
        }

        return -1;
    }
}

// 74. Search a 2D Matrix

// Day 2 Binary Search
// 153. Find Minimum in Rotated Sorted Array

public class Solution {
    public int FindMin(int[] nums) {
        int l = 0,  r = nums.Length - 1;
        
        while(l< r) {
 
        int m = l+(r-l)/2;
        if(nums[m] > nums[r])
            l = m + 1;
        else if (nums[m] < nums[r])
            r = m;
        }
        return nums[l];
    }
}

// 162. Find Peak Element
public class Solution {
    public int FindPeakElement(int[] nums) {
        int l = 0,  r = nums.Length - 1;
        
        while(l< r) {
 
        int m = l+(r-l)/2;
        if(nums[m] > nums[m+1]){r=m;}
        else if (nums[m] < nums[m+1]) {l=m+1;}
        }
        return l;
    }
}

public class Solution {
    public int FindPeakElement(int[] nums) {
        int l = 0,  r = nums.Length - 1;
        
        while(l< r) {
 
        int m = l+(r-l)/2;
        if(nums[m] > nums[m+1]){r=m;}
        else {l=m+1;}
        }
        return l;
    }
}

// Day 3 Two Pointers
// 82. Remove Duplicates from Sorted List II

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
    public ListNode DeleteDuplicates(ListNode head) {
         // sentinel
        ListNode sentinel = new ListNode(0, head);

        // predecessor = the last node 
        // before the sublist of duplicates
        ListNode pred = sentinel;
        
        while (head != null) {
            // if it's a beginning of duplicates sublist 
            // skip all duplicates
            if (head.next != null && head.val == head.next.val) {
                // move till the end of duplicates sublist
                while (head.next != null && head.val == head.next.val) {
                    head = head.next;    
                }
                // skip all duplicates
                pred.next = head.next;     
            // otherwise, move predecessor
            } else {
                pred = pred.next;    
            }
                
            // move forward
            head = head.next;    
        }  
        return sentinel.next;
    }
}

// 15. 3Sum
// Solution 2: Sorting + Two pointers
// Time complexity: O(nlogn + n^2)
// Space complexity: O(1)
public class Solution {
    public IList<IList<int>> ThreeSum(int[] nums) {
        HashSet<List<int>> res  = new HashSet<List<int>>();
        if(nums.Length==0) return new List<IList<int>>(res);
        Array.Sort(nums);
        for(int i=0; i<nums.Length-2;i++){
            //Since the array is sorted there won't be any chance the next entries sum to 0.
            if (nums[i] > 0) break; 
            // skip same result
            if (i > 0 && nums[i] == nums[i - 1]) continue; 
            int l = i+1;
            int r = nums.Length-1;
            while(l<r){ 
                if(nums[i]+nums[l]+nums[r]==0){
                    res.Add(new List<int>(){nums[i],nums[l++],nums[r--]});
                    // skip same result
                    while (l < r && nums[l] == nums[l - 1]) l++; 
                    // skip same result
                    while (l < r && nums[r] == nums[r + 1]) r--;
                }
                    
                else if (nums[i]+nums[l]+nums[r]>0) r--;
                else if (nums[i]+nums[l]+nums[r]<0) l++;
            }

        }
        return new List<IList<int>>(res);
    }
}

// Day 4 Two Pointers
// 844. Backspace String Compare
/*
Approach //2: Two Pointer
Time Complexity: O(M + N)O(M+N), where M, NM,N are the lengths of S and T respectively.
Space Complexity: O(1)O(1).
*/
public class Solution {
    public bool BackspaceCompare(string s, string t) {
        int i = s.Length - 1, j = t.Length - 1;
        int skipS = 0, skipT = 0;

        while (i >= 0 || j >= 0) { 
            // While there may be chars in build(S) or build (T)
            while (i >= 0) { 
                // Find position of next possible char in build(S)
                if (s[i] == '//') {skipS++; i--;}
                else if (skipS > 0) {skipS--; i--;}
                else break;
            }
            while (j >= 0) { 
                // Find position of next possible char in build(T)
                if (t[j] == '//') {skipT++; j--;}
                else if (skipT > 0) {skipT--; j--;}
                else break;
            }
            // If two actual characters are different
            if (i >= 0 && j >= 0 && s[i] != t[j])
                return false;
            // If expecting to compare char vs nothing
            if ((i >= 0) != (j >= 0))
                return false;
            i--; j--;
        }
        return true;
    }
}

// 986. Interval List Intersections
/*
Solution: Two pointers
Time complexity: O(m + n)
Space complexity: O(1)
*/
public class Solution {
    public int[][] IntervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> ans = new List<int[]>();
        int i = 0, j = 0;

        while (i < firstList.Length && j < secondList.Length) {
          // Let's check if A[i] intersects B[j].
          // l - the startpoint of the intersection
          // r - the endpoint of the intersection
          int l = Math.Max(firstList[i][0], secondList[j][0]);
          int r = Math.Min(firstList[i][1], secondList[j][1]);
          if (l <= r)
            ans.Add(new int[]{l, r});

          // Remove the interval with the smallest endpoint
          if (firstList[i][1] < secondList[j][1])
            i++;
          else
            j++;
        }

        return ans.ToArray();
    }
}

// 11. Container With Most Water
/*
Two pointers
Time complexity: O(n)
Space complexity: O(1)
*/
public class Solution {
    public int MaxArea(int[] height) {
        int l = 0;
        int r = height.Length - 1;
        int ans = 0;
        while (l < r) {
            int h = Math.Min(height[l], height[r]);
            ans = Math.Max(ans, h * (r - l));
            if (height[l] < height[r])
                ++l;
            else
                --r;
        }
        return ans;
    }
}

// Day 5 Sliding Window
// 438. Find All Anagrams in a String
/*
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
*/
public class Solution {
    public IList<int> FindAnagrams(string s, string p) {
        int[] cnt = new int[128];
        foreach (char c in p.ToCharArray()) cnt[c]++;
        
        List<int> ans = new List<int>();
        for (int r = 0, l = 0; r < s.Length; ++r) {
            char c = s[r];
            cnt[c]--;
            while (cnt[c] < 0) { // If number of characters `c` is more than our expectation
                cnt[s[l]]++;  // Slide left until cnt[c] == 0
                l++;
            }

            if (r - l + 1 == p.Length) { // If we already filled enough `p.length()` chars
                ans.Add(l); // Add left index `l` to our result
            }
        }
        return ans;
    }
}

// 713. Subarray Product Less Than K
// Approach 2: Sliding Window 
/*
The idea is that we keep 2 points l (initial value = 0) point to the left most of window, 
r point to current index of nums.
We use product (initial value = 1) to keep the product of numbers in the window range.
While iterating r in [0...n-1], we calculate number of subarray 
which ends at nums[r] has product less than k.
product *= nums[r].
While product > k && l <= r then we slide l by one
Now product < k, then there is r-l+1 subarray which ends at nums[r] has product less than k.
*/
// Complexity:
// Time: O(N), where N <= 3*10^4 is length of nums array.
// Space: O(1)
public class Solution {
    public int NumSubarrayProductLessThanK(int[] nums, int k) {
        if (k <= 1) return 0;
        int prod = 1, ans = 0, left = 0;
        for (int right = 0; right < nums.Length; right++) {
            prod *= nums[right];
            while (prod >= k) prod /= nums[left++];
            ans += right - left + 1;
        }
        return ans;
    }
}

// 209. Minimum Size Subarray Sum
/*
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
*/
public class Solution {
    public int MinSubArrayLen(int target, int[] nums) {
        
        int i = 0, n = nums.Length, res = n + 1;
        for (int j = 0; j < n; ++j) {
            target -= nums[j];
            while (target <= 0) {
                res = Math.Min(res, j - i + 1);
                target += nums[i++];
            }
        }
        return res % (n + 1);
    }
}
/*Approach 4 Using 2 pointers
Complexity
Time O(N)
Space O(1)
*/
public class Solution {
    public int MinSubArrayLen(int target, int[] nums) {
        
        int n = nums.Length;
        int ans = int.MaxValue;
        int left = 0;
        int sum = 0;
        for (int i = 0; i < n; i++) {
            sum += nums[i];
            while (sum >= target) {
                ans = Math.Min(ans, i + 1 - left);
                sum -= nums[left++];
            }
        }
        return (ans != int.MaxValue) ? ans : 0;
    }
}

// Day 6 Breadth-First Search / Depth-First Search
// 200. Number of Islands
/*
Idea: DFS
Use DFS to find a connected component (an island) and mark all the nodes to 0.

Time complexity: O(mn)
Space complexity: O(mn)
*/
public class Solution {
    public int NumIslands(char[][] grid) {
        int m = grid.Length;
        if (m == 0) return 0;
        int n = grid[0].Length;
        
        int ans = 0;
        for (int y = 0; y < m; ++y)
            for (int x = 0; x < n; ++x)
                if (grid[y][x] == '1') {
                    ++ans;
                    dfs(grid, x, y, n, m);
                }
        
        return ans;
    }
    
    private void dfs(char[][] grid, int x, int y, int n, int m) {
        if (x < 0 || y < 0 || x >= n || y >= m || grid[y][x] == '0')
            return;
        grid[y][x] = '0';
        dfs(grid, x + 1, y, n, m);
        dfs(grid, x - 1, y, n, m);
        dfs(grid, x, y + 1, n, m);
        dfs(grid, x, y - 1, n, m);
    }
}
// 547. Number of Provinces
// Solution: DFS
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
// Solution : Union Find
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

// Day 7 Breadth-First Search / Depth-First Search

// 117. Populating Next Right Pointers in Each Node II
// constant space
/*
// Definition for a Node.
public class Node {
    public int val;
    public Node left;
    public Node right;
    public Node next;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val, Node _left, Node _right, Node _next) {
        val = _val;
        left = _left;
        right = _right;
        next = _next;
    }
}
*/

public class Solution {
    public Node Connect(Node root) {
        Node dummyHead  = new Node(0); // this head will always point to the first element in the current layer we are searching
        Node pre = dummyHead; // this 'pre' will be the "current node" that builds every single layer   
        Node real_root = root; // just for return statement

        while(root != null){
          if(root.left != null){
              pre.next = root.left;
              pre = pre.next;
          }
          if(root.right != null){
              pre.next = root.right; 
              pre = pre.next;
          }
          root = root.next; 
          if(root == null){ // reach the end of current layer
              pre = dummyHead; // shift pre back to the beginning, get ready to point to the first element in next layer  
              root = dummyHead.next; ;//root comes down one level below to the first available non null node
              dummyHead.next = null;//reset dummyhead back to default null
          }
        }
        return real_root;
    }
}

// 572. Subtree of Another Tree

// tree traversal
// For each node during pre-order traversal of s, 
// use a recursive function isSame to validate if sub-tree started with this node is the same with t.
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
    public bool IsSubtree(TreeNode root, TreeNode subRoot) {
        if (root == null) return false;
        if (isSame(root, subRoot)) return true;
        return IsSubtree(root.left, subRoot) || IsSubtree(root.right, subRoot);
    }
    
    private bool isSame(TreeNode s, TreeNode t) {
        if (s == null && t == null) return true;
        if (s == null || t == null) return false;
        
        if (s.val != t.val) return false;
        
        return isSame(s.left, t.left) && isSame(s.right, t.right);
    }
}

// Day 8 Breadth-First Search / Depth-First Search

// 1091. Shortest Path in Binary Matrix
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

/*
DFS
The idea is to only go deeper if the dp value of one cell or its neibor is updated to a smaller value.
This is one trick for designing DFS: we shall try to gain in each recursion.
 The code actually let each cell c propagates between neighbors the length of currently found shortest path from source to c.
*/
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

// 130. Surrounded Regions
// DFS 
public class Solution {
    public void Solve(char[][] board) {
        if (board.Length == 0 || board[0].Length == 0) return;
		int m = board.Length;
		int n = board[0].Length;

		// go through the first column and the last column
		for (int i = 0; i < m; i++) {
			dfs(board, i, 0);
			dfs(board, i, n - 1);	
		}

		// go through the first row and the last row
		for (int j = 1; j < n - 1; j++) {
			dfs(board, 0, j);
			dfs(board, m - 1, j);	
		}

                // make all the remaining 'O' to 'X'
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O') board[i][j] = 'X';
                                if (board[i][j] == '*') board[i][j] = 'O';
			}
		}
	}


	// make every 'O' that we meet to '*' 
    // It is safe because we always start from the border
	public void dfs(char[][] board, int i, int j) {
		if (i < 0 || i >= board.Length || j < 0 || j >= board[0].Length) return;
		if (board[i][j] == 'X' || board[i][j] == '*') return;
		board[i][j] = '*';
		dfs(board, i - 1, j);
		dfs(board, i + 1, j);
		dfs(board, i, j - 1);
		dfs(board, i, j + 1);
    }
}

// BFS
public class Solution {
    public void Solve(char[][] board) {
        if (board.Length == 0 || board[0].Length == 0) return;
		int r = board.Length;
		int c = board[0].Length;
        if (r<=2 || c<=2) return;

        Queue<int[]> q = new Queue<int[]>();
		for (int i = 0; i < r; i++) {
			q.Enqueue(new int[]{i, 0});
            q.Enqueue(new int[]{i, c - 1});
		}

		for (int j = 1; j < c; j++) {
			q.Enqueue(new int[]{0, j});
            q.Enqueue(new int[]{r-1, j});
		}

         while (q.Count!=0){
             
            int[] n = q.Dequeue();
            int rn = n[0];
            int cn = n[1];
            // print(r, c)
            if (0 <= rn && rn < r && 0 <= cn && cn < c && board[rn][cn] == 'O'){
                board[rn][cn] = 'N';
                q.Enqueue(new int[]{rn-1, cn});
                q.Enqueue(new int[]{rn + 1, cn});
                q.Enqueue(new int[]{rn, cn + 1});
                q.Enqueue(new int[]{rn, cn - 1});
            }
                
         }
        
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				if (board[i][j] == 'N') board[i][j] = 'O';
                else board[i][j] = 'X';
			}
		}
	}   
}

// 797. All Paths From Source to Target
/* Backtracking
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
 

// Day 9 Recursion / Backtracking
// 78. Subsets

// Binary / Bit operation
// Time complexity: O(n * 2^n)
// Space complexity: O(1)
public class Solution {
    public IList<IList<int>> Subsets(int[] nums) {
        int n = nums.Length;
        List<IList<int>> ans = new List<IList<int>>();    
        for (int s = 0; s < 1 << n; ++s) {
          IList<int> cur = new List<int>();
          for (int i = 0; i < n; ++i)
            if ((s & (1 << i)) > 0) cur.Add(nums[i]);
          ans.Add(cur);
        }
        return ans;
    }
}

// DFS + Backtracking
// Time complexity:O(n * 2^n)
// Space complexity: O(n)
public class Solution {
    
    List<IList<int>> ans = new List<IList<int>>();
    IList<int> cur = new List<int>();
    
    public IList<IList<int>> Subsets(int[] nums) {
        
        for (int k = 0; k <= nums.Length; ++k) {
          dfs(nums, k, 0, cur, ans);
        }
        return ans;
    }
    
    public void dfs(int[] nums, int n, int s, IList<int> cur, List<IList<int>> ans) {
        // if the combination is done
        if (cur.Count == n) {
          ans.Add(new List<int>(cur));
          return;
        }
        for (int i = s; i < nums.Length; ++i) {
          // add i into the current combination
          cur.Add(nums[i]);
          // use next integers to complete the combination
          dfs(nums, n, i + 1, cur, ans);
          // backtrack
          cur.RemoveAt(cur.Count - 1);
        }
    }
}

// 90. Subsets II
/*
Solution: DFS
The key to this problem is how to remove/avoid duplicates efficiently.
For the same depth, among the same numbers, only the first number can be used.

Time complexity: O(2^n * n)
Space complexity: O(n)
*/
public class Solution {
    public IList<IList<int>> SubsetsWithDup(int[] nums) {
        int n = nums.Length;
        Array.Sort(nums);
        List<IList<int>> ans = new List<IList<int>>();
        IList<int> cur = new List<int>();
        
        void dfs(int s) {
            ans.Add(new List<int>(cur));
            if (cur.Count == n)
                return;      
            for (int i = s; i < n; ++i) {
                if (i > s && nums[i] == nums[i - 1]) continue;
                cur.Add(nums[i]);
                dfs(i + 1);
                cur.RemoveAt(cur.Count-1);
            }
        };
        
        dfs(0);
        return ans;
    }
}

public class Solution {
    public IList<IList<int>> SubsetsWithDup(int[] nums) {
        int n = nums.Length;
        Array.Sort(nums);
        List<IList<int>> ans = new List<IList<int>>();
        Stack<int> cur = new Stack<int>();
        
        void dfs(int s) {
          ans.Add(cur.ToList());
          if (cur.Count == n)
            return;      
          for (int i = s; i < n; ++i) {
            if (i > s && nums[i] == nums[i - 1]) continue;
            cur.Push(nums[i]);
            dfs(i + 1);
            cur.Pop();
          }
        };

        dfs(0);
        return ans;  
    }
}

public class Solution {
    
    List<IList<int>> ans = new List<IList<int>>();
    IList<int> cur = new List<int>();
    
    public void dfs(int s, int[] nums) {
          ans.Add(new List<int>(cur));
          if (cur.Count == nums.Length)
            return;      
          for (int i = s; i < nums.Length; ++i) {
            if (i > s && nums[i] == nums[i - 1]) continue;
            cur.Add(nums[i]);
            dfs(i + 1, nums);
            cur.RemoveAt(cur.Count-1);
          }
    }
    
    public IList<IList<int>> SubsetsWithDup(int[] nums) {
        Array.Sort(nums);       
        dfs(0, nums);
        return ans;
        
    }
}
// Day 10 Recursion / Backtracking

// 47. Permutations II
// Approach : Backtracking with Groups of Numbers
// Time Complexity: k-permutations_of_N or partial permutation.
// Space Complexity: O(N)
public class Solution {
    public IList<IList<int>> PermuteUnique(int[] nums) {
      List<IList<int>> results = new List<IList<int>>();

        // count the occurrence of each number
        Dictionary<int, int> counter = new Dictionary<int, int>();
        foreach (int num in nums) {
            if (!counter.ContainsKey(num))
                counter[num]= 0;
            counter[num] += 1;
        }

        IList<int> comb = new List<int>();
        this.backtrack(comb, nums.Length, counter, results);
        return results;
    }

    protected void backtrack(
            IList<int> comb,
            int N,
            Dictionary<int, int> counter,
            List<IList<int>> results) {

        if (comb.Count == N) {
            // make a deep copy of the resulting permutation,
            // since the permutation would be backtracked later.
            results.Add(comb.ToList()); // results.Add(new List<int>(comb)); would works too!
            return;
        }

        foreach (KeyValuePair< int, int> entry in counter) {
            int num = entry.Key;
            int count = entry.Value;
            if (count == 0)
                continue;
            // add this number into the current combination
            comb.Add(num);
            counter[num] = count - 1;

            // continue the exploration
            backtrack(comb, N, counter, results);

            // revert the choice for the next exploration
            comb.RemoveAt(comb.Count-1);
            counter[num] = count;
        }
    }
    
}

// 39. Combination Sum
// DFS
public class Solution {
    public IList<IList<int>> CombinationSum(int[] candidates, int target) {
        List<IList<int>> ans = new List<IList<int>>();
        IList<int> cur = new List<int>();
        Array.Sort(candidates);
        dfs(candidates, target, 0, cur, ans);
        return ans;
    }
    public void dfs(int[] candidates, int target, int s, IList<int> cur, List<IList<int>> ans) {
        if (target == 0) {
            ans.Add(new List<int>(cur));
            return;
        }
        
        for (int i = s; i < candidates.Length; ++i) {
            if (candidates[i] > target) break;
            cur.Add(candidates[i]);
            dfs(candidates, target - candidates[i], i, cur, ans);
            cur.RemoveAt(cur.Count-1);
        }
    }
}

// 40. Combination Sum II
// DFS
// Time complexity: O(2^n)
// Space complexity: O(kn)
// How to remove duplicates?
// 1. Use set
// 2. Disallow same number in same depth 

public class Solution {
    public IList<IList<int>> CombinationSum2(int[] candidates, int target) {
        List<IList<int>> ans = new List<IList<int>>();
        Array.Sort(candidates);
        IList<int> curr = new List<int>();
        dfs(candidates, target, 0, ans, curr);
        return ans;
    }
    public void dfs(int[] candidates, 
             int target, int s, 
             List<IList<int>> ans,              
             IList<int> curr) {
        if (target == 0) {
            ans.Add(new List<int>(curr));
            return;
        }
        
        for (int i = s; i < candidates.Length; ++i) {
            int num = candidates[i];
            if (num > target) return;
            if (i > s && candidates[i] == candidates[i - 1]) continue;
            curr.Add(num);
            dfs(candidates, target - num, i + 1, ans, curr);
            curr.RemoveAt(curr.Count-1);
        }
    }
}

// Day 11 Recursion / Backtracking

// 17. Letter Combinations of a Phone Number

// 22. Generate Parentheses
// DFS
// Solution: DFS
// Time complexity: O(2^n)
// Space complexity: O(k + n)
public class Solution {
    public IList<string> GenerateParenthesis(int n) {
        IList<string> ans = new List<string>();
        //string cur = string.Empty;
        if (n > 0) dfs(n, n, new StringBuilder(), ans);
        return ans;
    }
    public void dfs(int l, int r, StringBuilder s, IList<string> ans) {
        if (r < l) { return; }
        if (l + r == 0) {
            ans.Add(s.ToString());
            return;
        }
        
        if (l > 0) {   
            s.Append("(");
            dfs(l - 1, r,  s , ans);
            s.Remove(s.Length-1,1);
        }
        if (r > 0) {
            s.Append(")");
            dfs(l, r - 1,  s , ans);
            s.Remove(s.Length-1,1);
        }
    }
}

public class Solution {
    public IList<string> GenerateParenthesis(int n) {
        IList<string> ans = new List<string>();
        //string cur = string.Empty;
        if (n > 0) dfs(n, n, "", ans);
        return ans;
  }
    public void dfs(int l, int r, string s, IList<string> ans) {
        if (r < l) { return; }
        if (l + r == 0) {
            ans.Add(s);
            return;
        }
        
        if (l > 0) {   
            //s=s+"(";
            dfs(l - 1, r,  s+"(" , ans);
            //s.Remove(s.Length-1);
        }
        if (r > 0) {
            // s=s+")";
            dfs(l, r - 1, s+")", ans);
            //s.Remove(s.Length-1);
        }
    }
}

// 79. Word Search
// DFS
// Time complexity:O(m*n*4^l) l = len(word)
// Space complexity: O(m*n + l)
public class Solution {
    public int w;
    public int h;
    public bool Exist(char[][] board, string word) {
        if(board.Length==0) return false;
        h = board.Length;
        w = board[0].Length;
        for(int i=0;i<w;i++)
            for(int j=0;j<h;j++)
                if(search(board, word, 0, i, j)) return true;
        return false;
    }
    
    public bool search(char[][] board, 
             string word, int d, int x, int y) {
        //out of bound
        if(x<0 || x==w || y<0 || y==h || word[d] != board[y][x]) 
            return false;
        
        // Found the last char of the word
        if(d==word.Length-1)
            return true;
        
        char cur = board[y][x];
        board[y][x] = '-'; 
        bool found = search(board, word, d+1, x+1, y)
                  || search(board, word, d+1, x-1, y)
                  || search(board, word, d+1, x, y+1)
                  || search(board, word, d+1, x, y-1);
        board[y][x] = cur;
        return found;
    }

}

// Day 12 Dynamic Programming

// 213. House Robber II
public class Solution {
    public int Rob(int[] nums) {
        if(nums.Length<1) {return 0;}
        else if(nums.Length ==1) { return nums[0];}
        else
        {
            // arr[^0] means index after last element  arr[arr.Length] 
            // arr[^1] means last element equivalent to  [arr.Length - 1]
            // arr[^i] index from the end   reads:  arr[arr.Length -i]
            // range [1..4] returns {1, 2, 3 }
            // start of the range (1) is inclusive
            //   end of the range (4) is exclusive
            // [0..^0] means from the beginning till the end
            // It is equivalent to [..]
            // Remember that the upper bound ^0 is exclusive
            // so there is no risk of IndexOutOfRangeException here
            // [2..^2]   means  [2..(6-2)]  means  [2..4]
            return Math.Max(RobHelp(nums[0..^1]),RobHelp(nums[1..^0]));
        }
        
    }
    
    public int RobHelp(int[] nums)
    {   if (nums.Length == 0) return 0;
        int dp2 = 0;
        int dp1 = 0;
        for (int i = 0; i < nums.Length;i++) {
            int dp = Math.Max(dp2 + nums[i], dp1);
            dp2 = dp1;
            dp1 = dp;
        }
        return dp1;
    }
}

/* O(1)-Space
This problem is a little tricky at first glance. However, if you have finished the House Robber problem, this problem can simply be decomposed into two House Robber problems.
Suppose there are n houses, since house 0 and n - 1 are now neighbors, we cannot rob them together and thus the solution is now the maximum of

Rob houses 0 to n - 2;
Rob houses 1 to n - 1.
The code is as follows. Some edge cases (n < 2) are handled explicitly.*/
public class Solution {
    public int Rob(int[] nums) {
        if (nums.Length == 0) return 0;
        if (nums.Length == 1) return nums[0];
        return Math.Max(Rob(nums, 0, nums.Length - 2), Rob(nums, 1, nums.Length - 1));
    }
    
    public int Rob(int[] nums, int start, int end) 
    {   int prev1 = 0;
        int prev2 = 0;
        int curr = 0;
        for (int i = start; i <= end; i++)
        {
            curr = Math.Max(nums[i] + prev2, prev1);
            prev2 = prev1;
            prev1 = curr;
        }
        return curr;
    }
}

// 55. Jump Game
// Solution : Max Pos So Far
// Complexity
// Time: O(N), where N <= 10^4 is length of nums array.
// Space: O(1)
public class Solution {
    public bool CanJump(int[] nums) {
        int n = nums.Length, maxPos = 0,i = 0;
        while(i<=maxPos){
            maxPos = Math.Max(maxPos, i + nums[i]);
            if (maxPos >= n - 1) return true;
            i += 1;
        }
        return false;
    }
}

// Simplest O(N) solution with constant space
// Idea is to work backwards from the last index.
// Keep track of the smallest index that can "jump" to the last index.
// Check whether the current index can jump to this smallest index.
// The idea is: whenever we realize that we cannot reach a point i, return false.
public class Solution {
    public bool CanJump(int[] nums) {
        int maxLocation = 0;
        for(int i=0; i<nums.Length; i++) {
            if(maxLocation<i) return false; // if previous maxLocation smaller than i, meaning we cannot reach location i, thus return false.
            maxLocation = (i+nums[i]) > maxLocation ? i+nums[i] : maxLocation; // greedy:
        }
        return true;

    }
}

// Day 13 Dynamic Programming

// 45. Jump Game II
/*
Solution : Greedy
The main idea is based on greedy.
Step 1: Let's say the range of the current jump is [left, right], 
farthest is the farthest position that all positions in [left, right] can reach.
Step 2: Once we reach to right, we trigger another jump with left = right + 1, 
right = farthest, then repeat step 1 util we reach at the end.

Complexity
Time: O(N), where N <= 10^4 is the length of array nums
Space: O(1)
*/
public class Solution {
    public int Jump(int[] nums) {
        int jumps = 0, farthest = 0;
        int left = 0, right = 0;
        while (right < nums.Length - 1) {
            for (int i = left; i <= right; ++i)
                farthest = Math.Max(farthest, i + nums[i]);
            left = right + 1;
            right = farthest;
            ++jumps;
        }
        return jumps;
    }
}
/*
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
*/
public class Solution {
    public int Jump(int[] nums) {
        int len = nums.Length - 1, curr = -1, next = 0, ans = 0;
        for (int i = 0; next < len; i++) {
            if (i > curr) {
                ans++;
                curr = next;
            };
            next = Math.Max(next, nums[i] + i);
        };
        return ans;
    }
}

// 62. Unique Paths
// DP
public class Solution {  
    public int UniquePaths(int m, int n) {
        int[][] grid = new int[m][];
        for(int i = 0; i<m; i++){
            grid[i] = new int[n];
            for(int j = 0; j<n; j++){
                if(i==0||j==0)
                    grid[i][j] = 1;
                else
                    grid[i][j] = grid[i][j-1] + grid[i-1][j];
            }
        }
        return grid[m-1][n-1];
    } 
}

//DP
public class Solution {  
    public int UniquePaths(int m, int n) {
        int[][] grid = new int[n+1][];
        for(int i = 0; i<= n; i++){ grid[i] = new int[m+1]; Array.Fill(grid[i], 0);}
        for(int i = 1; i<= n; i++){
            for(int j = 1; j<= m; j++){
                if(i==1 && j==1){
                    grid[1][1] = 1;
                    continue;
                } else {
                    grid[i][j] = grid[i][j-1] + grid[i-1][j];
                }
            }
        }
        return grid[n][m];

    } 
}

//DFS
public class Solution {
    private int[][] dp;
    private int m;
    private int n;
  
    public int UniquePaths(int m, int n) {
        this.dp = new int[m][];    
        this.m = m;
        this.n = n;
        for(int i=0;i<m;i++){dp[i]=new int[n];}
        return dfs(0, 0);
    }
    
    private int dfs(int x, int y) {
        if (x > m - 1 || y > n - 1)
          return 0;
        if (x == m - 1 && y == n - 1)
          return 1;
        if (dp[x][y] == 0)     
          dp[x][y] = dfs(x + 1, y) + dfs(x , y + 1);
        return dp[x][y];
    } 
}

// Day 14 Dynamic Programming

// 5. Longest Palindromic Substring
/* DP
 dp(i, j) represents whether s(i ... j) can form a palindromic substring, 
 dp(i, j) is true when s(i) equals to s(j) and s(i+1 ... j-1) is a palindromic substring. 
 When we found a palindrome, check if it's the longest one. 
 Time complexity O(n^2).

j must be greater than or equal i at all times. Why? 
i is the start index of the substring, j is the end index of the substring.
It makes no sense for i to be greater than j. 
Visualization helps me, so if you visualize the dp 2d array, 
think of a diagonal that cuts from top left to bottom right. 
We are only filling the top right half of dp.

Why are we counting down for i, but counting up for j? 
Each sub-problem dp[i][j] depends on dp[i+1][j-1] 
(dp[i+1][j-1] must be true and s[i] must equal s[j] for dp[i][j] to be true).
*/
public class Solution {
    public string LongestPalindrome(string s) {
        int n = s.Length;
        String res = null;
            
        // dp[i][j] indicates whether substring s starting at index i and ending at j is palindrome    
        bool[][] dp = new bool[n][];
        for (int i = 0; i < n; i++) { dp[i] = new bool[n];}
            
        for (int i = n - 1; i >= 0; i--) { // keep increasing the possible palindrome string
            for (int j = i; j < n; j++) { // find the max palindrome within this window of (i,j)
                //check if substring between (i,j) is palindrome
                dp[i][j] = ( s[i] == s[j])  // chars at i and j should match
                            && 
                            (j - i < 3 // if window is less than or equal to 3, just end chars should match
                                || dp[i + 1][j - 1]); // if window is > 3, substring (i+1, j-1) should be palindrome too
                 //update max palindrome string        
                if (dp[i][j] == true && (res == null || j - i + 1 > res.Length)){
                    res = s.Substring(i, j + 1-i);
                }
            }
        }
            
        return res;
    }
}


// 413. Arithmetic Slices
/*
i) We need minimum 3 indices to make arithmetic progression,
ii) So start at index 2, see if we got two diffs same, so we get a current 1 arith sequence
iii) At any index i, if we see it forms arith seq with former two, that means running (curr) sequence gets extended upto this index, at the same time we get one more sequence (the three numbers ending at i), so curr++. Any time this happens, add the curr value to total sum.
iv) Any time we find ith index does not form arith seq, make currently running no of seqs to zero.
*/
public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        int curr = 0, sum = 0;
        for (int i=2; i<nums.Length; i++)
            if (nums[i]-nums[i-1] == nums[i-1]-nums[i-2]) {
                curr += 1;
                sum += curr; //adding current number to our existing arithmetic sequence, we will have curr additional combinations of new arithmetic slices.
            } else {
                curr = 0;
            }
        return sum;
    }
}

/*
An Arithmetic Slice (AS) is at least 3 ints long, s.t. for a1, a2, a3, they're in arithmetic progression (AP), i.e. a3 - a2 = a2 - a1
If there's an AS forming at any index, then it'll be 1 longer than the AS forming at the previous index. Why?
Let a[i], ..., a[j] form an AS of size k, and
a[j + 1] - a[j] = a[j] - a[j - 1], then a[j + 1] becomes a part of the previous AS, i.e. it extends the AS by 1
Total AS = sum of all count of AS ending at each index
Example: nums: [1, 2, 3, 8, 9, 10]

Let AS[i] denote number of AS ending at this index. 
By definition AS[0] = AS[1] = 0. Also, AS[i] = 0 for any index for which the current int and previous two ints are not in AP

nums 1 2 3 8 9 10
AS   0 0 1 0 0 1

since 3 - 2 = 2 - 1 ⇒ AS[2] = 1 + AS[1] = 1
8 - 3 ≠ 3 - 2 ⇒ AS[3] = 0
9 - 8 ≠ 8 - 3 ⇒ AS[4] = 0 
10 - 9 = 9 - 8 ⇒ AS[5] = 1 + AS[4] = 1

Total AS = 1 + 1 = 2 [Ans]
T/S: O(n)/O(1), where n = size(nums)
*/
public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        var slices = 0;
        for (int i = 2, prev = 0; i < nums.Length; i++)
            slices += (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) ? 
                    ++prev : 
                    (prev = 0);
        return slices;
    }
}

public class Solution {
    public int NumberOfArithmeticSlices(int[] nums) {
        var slices = 0;
        for (int i = 2, prev = 0; i < nums.Length; i++){
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) 
            {
                prev = prev + 1;
                slices = slices + prev;
            } 
            else {
                prev = 0;
            }
        }
        return slices;
    }
}
// Day 15 Dynamic Programming

// 91. Decode Ways
/* Solution : Bottom-up DP (Space Optimized)
Since our dp only need to keep up to 3 following states:
    Current state, let name dp corresponding to dp[i]
    Last state, let name dp1 corresponding to dp[i+1]
    Last twice state, let name dp2 corresponding to dp[i+2]
So we can achieve O(1) in space.

Complexity
Time: O(N), where N <= 100 is length of string s.
Space: O(1)
*/

public class Solution {
    public int NumDecodings(string s) {
        int n = s.Length, dp = 0, dp1 = 1, dp2 = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s[i] != '0') // Single digit
                dp += dp1;
            if (i+1 < s.Length && (s[i] == '1' || s[i] == '2' && s[i+1] <= '6')) // Two digits
                dp += dp2;
            dp2 = dp1;
            dp1 = dp;
            dp = 0;
        }
        return dp1;
    }
}

// 139. Word Break
/*
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
*/
public class Solution {
    public bool WordBreak(string s, IList<string> wordDict) {
        // Create a hashset of words for fast query.
        List<String> dict = new List<String>(wordDict);
        Dictionary<String, bool> mem = new Dictionary<String, bool>();
        return wordBreak(s, mem, dict);// Query the answer of the original string.
    }
 
    private bool wordBreak(String s,
                              Dictionary<String, bool> mem, 
                             List<String> dict) {
        if (mem.ContainsKey(s)) return mem[s];
        if (dict.Contains(s)) {// In memory, directly return.
            mem[s]= true;
            return true;
        }
        // Try every break point.
        for (int i = 1; i < s.Length; ++i) {
            // Find the solution for s.
            if (dict.Contains(s.Substring(i)) && wordBreak(s.Substring(0, i), mem, dict)) {
                mem[s]= true;
                return true;
            }
        }
        // No solution for s, memorize and return.
        mem[s]= false;
        return false;
    }
}

// Day 16 Dynamic Programming

// 300. Longest Increasing Subsequence

/* Solution : DP + Binary Search / Patience Sorting

Patience Sorting
 It might be easier for you to understand how it works if you think about it as piles of cards instead of tails. 
 The number of piles is the length of the longest subsequence. 

dp[i] := smallest tailing number of a increasing subsequence of length i + 1.
dp is an increasing array, we can use binary search to find the index to insert/update the array.
ans = len(dp)

Time complexity: O(nlogn)
Space complexity: O(n)
*/
public class Solution {
    public int LengthOfLIS(int[] nums) {
        List<int> dp = new List<int>();
        
        foreach (int x in nums) {
            int it = dp.BinarySearch(x);
            if (it < 0) { it = ~it;} // ~x : reversing each bit
            if (it == dp.Count){
                dp.Add(x);   
            }
            else{
                dp[it] = x;
            }
        }
        return dp.Count;
    }
}

/* Binary search O(nlogn) time

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
*/
public class Solution {
    public int LengthOfLIS(int[] nums) {
        int[] tails = new int[nums.Length];
    int size = 0;
    foreach (int x in nums) {
        int i = 0, j = size;
        while (i != j) {
            int m = (i + j) / 2;
            if (tails[m] < x)
                i = m + 1;
            else
                j = m;
        }
        tails[i] = x;
        if (i == size) ++size;
    }
    return size;
    }
}

// 673. Number of Longest Increasing Subsequence
/* Simple DP
The idea is to use two arrays len[n] and cnt[n] to record the maximum length of Increasing Subsequence and the coresponding number of these sequence which ends with nums[i], respectively. That is:

len[i]: the length of the Longest Increasing Subsequence which ends with nums[i].
cnt[i]: the number of the Longest Increasing Subsequence which ends with nums[i].

Then, the result is the sum of each cnt[i] while its corresponding len[i] is the maximum length.
*/
public class Solution {
    public int FindNumberOfLIS(int[] nums) {
        int n = nums.Length, res = 0, max_len = 0;
        int[] len =  new int[n], cnt = new int[n];
        for(int i = 0; i<n; i++){
            len[i] = cnt[i] = 1;
            for(int j = 0; j <i ; j++){
                if(nums[i] > nums[j]){
                    if(len[i] == len[j] + 1)cnt[i] += cnt[j];
                    if(len[i] < len[j] + 1){
                        len[i] = len[j] + 1;
                        cnt[i] = cnt[j];
                    }
                }
            }
            if(max_len == len[i])res += cnt[i];
            if(max_len < len[i]){
                max_len = len[i];
                res = cnt[i];
            }
        }
        return res;
    }
}

public class Solution {
    public int FindNumberOfLIS(int[] nums) {
        int n = nums.Length, ans = 0, max_len = 0;
        if(n == 0) return 0;
        int[] l =  new int[n], c = new int[n];
        Array.Fill(l,1); // Fill array with same element
        Array.Fill(c,1);
        for(int i = 1; i<n; i++){
            for(int j = 0; j <i ; j++){
                if(nums[i] > nums[j]){
                    if(l[i] == l[j] + 1) c[i] += c[j];
                    else if(l[i] < l[j] + 1){
                        l[i] = l[j] + 1;
                        c[i] = c[j];
                    }
                }
            }
        }
            max_len = l.Max(); // Max Value of an array
            for(int i = 0; i<n; i++){
                if(max_len == l[i]) ans += c[i];
            }
        
        return ans;
    }
}

// Day 17 Dynamic Programming

// 1143. Longest Common Subsequence

/*
Solution: DP

Use dp[i][j] to represent the length of longest common sub-sequence of text1[0:i] and text2[0:j]
dp[i][j] = dp[i – 1][j – 1] + 1 if text1[i – 1] == text2[j – 1] else max(dp[i][j – 1], dp[i – 1][j])

Time complexity: O(mn)
Space complexity: O(mn) -> O(n)
*/

public class Solution {
    public int LongestCommonSubsequence(string text1, string text2) {
        int m = text1.Length;
        int n = text2.Length;    
        int[] dp1 = new int[n+1] , dp2 = new int[n+1];    
        for (int i = 0; i < m; ++i) {
         
          for (int j = 0; j < n; ++j) {        
            
            if (text1[i] == text2[j])
              // dp[i + 1][j + 1] = dp[i][j] + 1
              dp2[j + 1] = dp1[j] + 1; 
            else
              // dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
              dp2[j + 1] = Math.Max(dp1[j + 1], dp2[j]);
            
          }    
            
            int[] temp = dp1;
            dp1 = dp2;
            dp2 = temp; 
        }    
        return dp1[n]; // dp[m][n]

    
    }
}

public class Solution {
    public int LongestCommonSubsequence(string text1, string text2) {
        int m = text1.Length;
        int n = text2.Length;    
        int[] dp = new int[n+1];    
        for (int i = 0; i < m; ++i) {
          int prev = 0; // dp[i][j]
          for (int j = 0; j < n; ++j) {        
            int curr = dp[j + 1]; // dp[i][j + 1]
            if (text1[i] == text2[j])
              // dp[i + 1][j + 1] = dp[i][j] + 1
              dp[j + 1] = prev + 1; 
            else
              // dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
              dp[j + 1] = Math.Max(curr, dp[j]);
            prev = curr;
          }    
        }    
        return dp[n]; // dp[m][n]
    }
}

// 583. Delete Operation for Two Strings
/*
Approach 1-D Dynamic Programming [Accepted]:
Complexity Analysis

Time complexity : O(m*n). 
We need to fill in the dpdp array of size nn, mm times. 
Here, mm and nn refer to the lengths of s1s1 and s2s2.

Space complexity : O(n). dp array of size nn is used.
*/
public class Solution {
    public int MinDistance(string word1, string word2) {
        int[] dp = new int[word2.Length + 1];
        for (int i = 0; i <= word1.Length; i++) {
            int[] temp = new int[word2.Length+1];
            for (int j = 0; j <= word2.Length; j++) {
                if (i == 0 || j == 0)
                    temp[j] = i + j;
                else if (word1[i - 1] == word2[j - 1])
                    temp[j] = dp[j - 1];
                else
                    temp[j] = 1 + Math.Min(dp[j], temp[j - 1]);
            }
            dp=temp;
        }
        return dp[word2.Length];
    }
}
// Day 18 Dynamic Programming

// 72. Edit Distance

// Iterative Dynamic Programming
public class Solution {
    public int MinDistance(string word1, string word2) {
        int l1 = word1.Length;
        int l2 = word2.Length;
        // d[i][j] := minDistance(word1[0:i - 1], word2[0:j - 1]);
        int[][] d = new int[l1+1][];
        
        for (int i = 0; i <= l1; ++i){
             d[i] = new int[l2+1];
            d[i][0] = i;
        }
           
        for (int j = 0; j <= l2; ++j){
            d[0][j] = j;
        }
            
        for (int i = 1; i <= l1; ++i){
            for (int j = 1; j <= l2; ++j) {
                int c = (word1[i - 1] == word2[j - 1]) ? 0 : 1;
                d[i][j] = Math.Min(d[i - 1][j - 1] + c, 
                              Math.Min(d[i][j - 1], d[i - 1][j]) + 1);
            }
        }
                   
        return d[l1][l2];
    }
}
// Recursive Dynamic Programming
public class Solution {
    public int MinDistance(string word1, string word2) {
        int l1 = word1.Length;
        int l2 = word2.Length;
        d_ = new int[l1 + 1][];
        for(int i=0; i<l1+1; i++){ d_[i] = new int[l2 + 1] ; Array.Fill(d_[i],-1);}
        return minDistance(word1, word2, l1, l2);
    }
    
    private int[][] d_;
    // minDistance from word1[0:l1-1] to word2[0:l2-1]
    public int minDistance(string word1, string word2, int l1, int l2) {
        if (l1 == 0) return l2;
        if (l2 == 0) return l1;
        if (d_[l1][l2] >= 0) return d_[l1][l2];
        
        int ans;
        if (word1[l1 - 1] == word2[l2 - 1])
            ans = minDistance(word1, word2, l1 - 1, l2 - 1);
        else 
            ans = Math.Min(minDistance(word1, word2, l1 - 1, l2 - 1),
                  Math.Min(minDistance(word1, word2, l1 - 1, l2), 
                      minDistance(word1, word2, l1, l2 - 1))) + 1;
        
        return d_[l1][l2] = ans;        
    }
}

// 322. Coin Change
/*
Solution : DP
Time complexity: O(n*amount)
Space complexity: O(amount)
*/
public class Solution {
    public int CoinChange(int[] coins, int amount) {
        int[] dp = Enumerable.Repeat(Int32.MaxValue, amount+1).ToArray();
        
        dp[0] = 0;
        foreach (int coin in coins) {
            for (int i = coin; i <= amount; ++i)
                if (dp[i - coin] != Int32.MaxValue)  
                    dp[i] = Math.Min(dp[i], dp[i - coin] + 1);
        }
        
        return dp[amount] == Int32.MaxValue ? -1 : dp[amount];
    }
}

/*
Solution : DFS+Greedy+Pruning
Use largest and as many as coins first to reduce the search space
Time complexity: O(amount^n/(coin_0*coin_1*…*coin_n))
Space complexity: O(n)
This one is TLE on LeetCode for C// Idk why
*/
public class Solution {
    public int CoinChange(int[] coins, int amount) {
        // Sort coins in desending order        
        //Array.Sort(coins);// Sort array in ascending order.
        //Array.Reverse(coins); // reverse array
        // Sort the array in decreasing order and return a array
        coins = coins.OrderByDescending(c => c).ToArray();
        //ans = Int32.MaxValue;
        coinChange(coins, 0, amount, 0);
        return ans == Int32.MaxValue ? -1 : ans;
    }
int ans = Int32.MaxValue;
public void coinChange(int[] coins, int s, int amount, int count) {
        if (amount == 0) {
            ans = Math.Min(ans, count);
            return;
        }
        
        if (s == coins.Length) return;
        
       int coin = coins[s];

        for (int k = amount / coin; k >= 0 && count + k < ans; k--){
            coinChange(coins, s+1, amount - k*coin, count + k);
        }
            
    }    
}
// 343. Integer Break

public class Solution {
    public int IntegerBreak(int n) {
        if(n > 3) n++;
        int[] dp = new int[n+1];
        dp[1] = 1;
        for(int i = 2; i <=n; i++) {
            for(int j = 1; j < i; j++) {
                dp[i] = Math.Max(dp[i], j * dp[i-j]);
            }
        }
        return dp[n];
    }
}

public class Solution {
    public int IntegerBreak(int n) {
        //dp[i] means output when input = i, e.g. dp[4] = 4 (2*2),dp[8] = 18 (2*2*3)...
        int[] dp = new int[n + 1];
        dp[1] = 1;
       // fill the entire dp array
        for (int i = 2; i <= n; i++) {
            //let's say i = 8, we are trying to fill dp[8]:
            //if 8 can only be broken into 2 parts, 
            //the answer could be among 1 * 7, 2 * 6, 3 * 5, 4 * 4... 
            //but these numbers can be further broken. 
            //so we have to compare 1 with dp[1], 7 with dp[7], 
            //2 with dp[2], 6 with dp[6]...etc
            for (int j = 1; j <= i / 2; j++) {
               // use Math.max(dp[i],....)  so dp[i] maintain the greatest value
                dp[i] = Math.Max(dp[i],Math.Max(j, dp[j]) * Math.Max(i - j, dp[i - j]));
            }
        }
        return dp[n];
    }
}

// Day 19 Bit Manipulation

// 201. Bitwise AND of Numbers Range
/*
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
*/
public class Solution {
    public int RangeBitwiseAnd(int left, int right) {
        int i = 0; // i means we have how many bits are 0 on the right
        while(left != right){
            left >>= 1;
            right >>= 1;
            i++;  
      }  
        return left << i;  
    }
}

// uint x = 0b_1001;
// Console.WriteLine($"Before: {Convert.ToString(x, toBase: 2), 4}");
// uint y = x >> 2;
// Console.WriteLine($"After:  {Convert.ToString(y, toBase: 2), 4}");
// Output:
// Before: 1001
// After:    10

// uint x = 0b_1100_1001_0000_0000_0000_0000_0001_0001;
// Console.WriteLine($"Before: {Convert.ToString(x, toBase: 2)}");
// uint y = x << 4;
// Console.WriteLine($"After:  {Convert.ToString(y, toBase: 2)}");
// Output:
// Before: 11001001000000000000000000010001
// After:  10010000000000000000000100010000

// Day 20 Others

// 384. Shuffle an Array
// Knuth Shuffle.
// In iteration i, pick integer r between 0 and i uniformly at random.
// Swap a[i] and a[r].
public class Solution {

    private int[] nums;
    
    public Solution(int[] nums) {
        this.nums = nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] Reset() {
        return nums;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] Shuffle() {
        int[] rand = new int[nums.Length];
        for (int i = 0; i < nums.Length; i++){
            //Random rand = new Random(); 
            //int r = rand.Next(i+1);
            //Math.random returns a positive double value >= 0.0 and < 1.0.
            int r = new Random().Next(i+1);//maxValue parameter
            rand[i] = rand[r];
            rand[r] = nums[i];
        }
        return rand;
    }
}

/**
 * Your Solution object will be instantiated and called as such:
 * Solution obj = new Solution(nums);
 * int[] param_1 = obj.Reset();
 * int[] param_2 = obj.Shuffle();
 */

// Day 21 Others

// 202. Happy Number
// An Alternative Implementation
// Thanks @Manky for sharing this alternative with us!

// This approach was based on the idea that all numbers either end at 1 or 
// enter the cycle {4, 16, 37, 58, 89, 145, 42, 20}, wrapping around it infinitely.

// An alternative approach would be to recognise that all numbers will either end at 1,
// or go past 4 (a member of the cycle) at some point.
// Therefore, instead of hardcoding the entire cycle, we can just hardcode the 4.
public class Solution {
    public bool IsHappy(int n) {
        while (n != 1 && n != 4) {
            n = getNext(n);
        }
        return n == 1;
    }
    
     public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }
}

/*
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
*/
public class Solution {
    public bool IsHappy(int n) {
        int slowRunner = n;
        int fastRunner = getNext(n);
        while (fastRunner != 1 && slowRunner != fastRunner) {
            slowRunner = getNext(slowRunner);
            fastRunner = getNext(getNext(fastRunner));
        }
        return fastRunner == 1;
    }
    
     public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }
}

/*
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
*/
public class Solution {
    
    private static List<int> cycleMembers =
        new List<int>(){4, 16, 37, 58, 89, 145, 42, 20};
    public int getNext(int n) {
        int totalSum = 0;
        while (n > 0) {
            int d = n % 10;
            n = n / 10;
            totalSum += d * d;
        }
        return totalSum;
    }

    public bool IsHappy(int n) {
        while (n != 1 && !cycleMembers.Contains(n)) {
            n = getNext(n);
        }
        return n == 1;
    }
}

// 149. Max Points on a Line

public class Solution {
    public int MaxPoints(int[][] points) {
        if(points.Length <= 0) return 0;
        if(points.Length <= 2) return points.Length;
        int result = 0;
        for(int i = 0; i < points.Length; i++){
            Dictionary<decimal, int> hm = new Dictionary<decimal, int>(); //Dictionary slope : count
            int samex = 1; 
            int samep = 0;// overlap point
            for(int j = i+1; j < points.Length; j++){
                if((points[j][0] == points[i][0]) && (points[j][1] == points[i][1])){
                        samep++;
                }
                if(points[j][0] == points[i][0]){
                    samex++;
                    continue;
                }
                // Use decimal not float for higher precision while dividing
                decimal k = (decimal)(points[j][1] - points[i][1]) / (decimal)(points[j][0] - points[i][0]);
                if(hm.ContainsKey(k)){
                    hm[k] += 1;
                }
                else{
                    hm.Add(k, 2);
                }
                result = Math.Max(result, hm[k] + samep);
            }
            result = Math.Max(result, samex);
        }
        return result;
    }
}

 /*
     *  A line is determined by two factors,say y=ax+b
     *  If two points(x1,y1) (x2,y2) are on the same line(Of course). 
     *  Consider the gap between two points.
     *  We have (y2-y1)=a(x2-x1),a=(y2-y1)/(x2-x1) a is a rational, b is canceled since b is a constant
     *  If a third point (x3,y3) are on the same line. So we must have y3=ax3+b
     *  Thus,(y3-y1)/(x3-x1)=(y2-y1)/(x2-x1)=a
     *  Since a is a rational, there exists y0 and x0, y0/x0=(y3-y1)/(x3-x1)=(y2-y1)/(x2-x1)=a
     *  So we can use y0 & x0 to track a line;
     */
public class Solution {
    public int MaxPoints(int[][] points) {
        if (points == null) return 0;
        
        int solution = 0;
        
        for (int i = 0; i < points.Length; i++)
        {
            Dictionary<string, int> map = new Dictionary<string, int>();
            int duplicate = 0;
            int max = 0;
            for (int j = i + 1; j < points.Length; j++)
            {
                int deltaX = points[j][0] - points[i][0];
                int deltaY = points[j][1] - points[i][1];
                
                if (deltaX == 0 && deltaY == 0)
                {
                    duplicate++;
                    continue;
                }
                
                int GCD = gcd(deltaX, deltaY);
                int dX = deltaX / GCD;
                int dY = deltaY / GCD;
                
                map[dX + "," + dY] =map.GetValueOrDefault(dX + "," + dY, 0) + 1;
                max = Math.Max(max, map[dX + "," + dY]);
            }
            
            solution = Math.Max(solution, max + duplicate + 1);
        }
        
        return solution;
    }

    public int gcd(int a, int b)
    {
        if (b == 0)
            return a;
        return gcd(b, a%b);
    }

}