//LeetCode Data Structure 
//Day 1 Array
//217. Contains Duplicate
//Array solution
public class Solution {
    public bool ContainsDuplicate(int[] nums) {
         Array.Sort(nums);//sort it first using array
        //then figure out whether if it's the same with next one
          for(int i=0; i<nums.Length-1; i++)
          {
                if(nums[i] == nums[i+1]) 
                    return true;
          }
      
        return false;
 
    }
}
// hash set solution
// A HashSet, similar to a Dictionary, is a hash-based collection,
// so look ups are very fast with O(1). But unlike a dictionary,
// it doesn't store key/value pairs; it only stores values.
// So, every objects should be unique and this is determined by
// the value returned from the GetHashCode method.
public class Solution {
    public bool ContainsDuplicate(int[] nums) {
        HashSet<int> set = new HashSet<int> ();
        int Length = nums.Length;
        if(Length <=1) return false;

            for(int i=0;  i<Length; i++)
            {
              if(set.Contains(nums[i]))
              {
                return true;   
              }
              else
              {        
                set.Add(nums[i]);
              }
            }

            return false;
        
    }
}

//53. Maximum Subarray
public class Solution {
    public int MaxSubArray(int[] nums) {
            int currentMax = nums[0];
            int globalMax = nums[0];
            for(int i = 1; i< nums.Length; ++i)
            {
                currentMax = Math.Max(nums[i], nums[i] + currentMax);
                globalMax = Math.Max(currentMax, globalMax);
            }

            return globalMax;
    }
}

public class Solution {
    public int MaxSubArray(int[] nums) {
           for (int i=1;i<nums.Length;i++)
           {
               if (nums[i-1] >0){
                   nums[i] += nums[i-1];
               }
                    
           }
        //return the max value of nums list
        return nums.Max();
    }
}

public class Solution {
    public int MaxSubArray(int[] nums) {
        int sum = nums[0];
	    int maxSum = nums[0];

        for (int i=1; i<nums.Length; i++) {
            
            if ( sum >0 ) {
                sum += nums[i];
            }
            else{
                sum = nums[i];
            }
            
            maxSum = Math.Max(sum, maxSum); 
        
        }
        return maxSum;
    }
}
// Day 2 Array
// 1. Two Sum
public class Solution {
    public int[] TwoSum(int[] nums, int target) {
        Dictionary<int, int> dict = 
                    new Dictionary<int, int>();
        for(int i =0; i<nums.Length;i++)
        {
            if(dict.ContainsKey(target-nums[i]))
            {
                return new int[]{dict[target-nums[i]],i};
            }
            dict[nums[i]] = i;
        }
        return new int[]{0,0};
    }
}

// 88. Merge Sorted Array
public class Solution {
    public void Merge(int[] nums1, int m, int[] nums2, int n) {
        while (n>0)
        {
            if(m<1 || nums1[m-1]<=nums2[n-1])
            {
                nums1[m+n-1] = nums2[n-1];
                n--;
            }
            else{
                nums1[m+n-1] = nums1[m-1];
                m--;
            }
        }
    }
}

public class Solution {
    public void Merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1, j = n - 1, k = m + n - 1;
     while(i >= 0 && j >= 0) 
     {
         if(nums1[i] < nums2[j]) {
             nums1[k--] = nums2[j--];
         } else {
             nums1[k--] = nums1[i--];
         }
     }
     while(j >= 0) {
         nums1[k--] = nums2[j--];
     }
    }
}

// Day 3 Array
// 350. Intersection of Two Arrays II
//two-pointer
public class Solution {
    public int[] Intersect(int[] nums1, int[] nums2) {
        Array.Sort(nums1);
        Array.Sort(nums2);
        int i = 0, j = 0;
        List<int> list = new List<int>();
        while(i < nums1.Length && j < nums2.Length){
            if(nums1[i] == nums2[j]){
                list.Add(nums1[i]);
                i++;
                j++;
            }
            else if(nums1[i] < nums2[j]){
                i++;
            }
            else{
                j++;
            }
        }
        return list.ToArray();
    }
}
/*
Follow Up:
Q:What if the given array is already sorted? How would you optimize your algorithm?

Classic two-pointer iteration, i points to nums1 and j points to nums2.
Because a sorted array is in ascending order, so if nums1[i] > nums[j], we need to increment j,
and vice versa. Only when nums1[i] == nums[j], we add it to the result array.
Time Complexity O(max(N, M)). Worst case, for example, would be nums1 = {100},
and nums2 = {1, 2, ..., 100 }. We will always iterate the longest array.

Q:What if nums1's size is small compared to nums2's size? Which algorithm is better?

This one is a bit tricky. Let's say nums1 is K size. 
Then we should do binary search for every element in nums1. 
Each lookup is O(log N), and if we do K times, we have O(K log N).
If K this is small enough, O(K log N) < O(max(N, M)). 
Otherwise, we have to use the previous two pointers method.
let's say A = [1, 2, 2, 2, 2, 2, 2, 2, 1], B = [2, 2]. 
For each element in B, we start a binary search in A. 
To deal with duplicate entry, once you find an entry, 
all the duplicate element is around that that index, 
so you can do linear search scan afterward.
Time complexity, O(K(logN) + N). 
Plus N is worst case scenario which you have to linear scan every element in A. 
But on average, that shouldn't be the case. 
so I'd say the Time complexity is O(K(logN) + c), c (constant) is number of linear scan you did.

Q:What if elements of nums2 are stored on disk, 
and the memory is limited such that you cannot load all elements into the memory at once?

If only nums2 cannot fit in memory, put all elements of nums1 into a HashMap,
read chunks of array that fit into the memory, and record the intersections.
If both nums1 and nums2 are so huge that neither fit into the memory,
sort them individually (external sort), then read 2 elements from each array at a time in memory, record intersections.
This one is open-ended. But Map-Reduce I believe is a good answer.
*/

// 121. Best Time to Buy and Sell Stock
// DP
public class Solution {
    public int MaxProfit(int[] prices) {
        if (prices.Length <=1)
        {
            return 0;
        }
        int max_profit = 0, min_price = prices[0];
        for(int i=0;i<prices.Length;i++)
        {
            max_profit = Math.Max(max_profit,prices[i]-min_price);
            min_price = Math.Min(min_price,prices[i]);
        }
        return max_profit;
    }
}

// Day 4 Array
// 566. Reshape the Matrix
public class Solution {
    public int[][] MatrixReshape(int[][] mat, int r, int c) {
        int n = mat.Length, m = mat[0].Length;
        if(r*c != n*m) return mat;
        /*While a true 2D array would have m rows of n elements each,
         a jagged array could have m rows each having different numbers of elements.*/
        int[][] result = new int[r][];
        for(int i = 0; i < r*c;i++)
        {
            if(i%c == 0)// add new row when it's zero
            {
                 result[i/c] = new int[c];
            }
            result[i/c][i%c] = mat[i/m][i%m];
            
        }
        return result;
    }
}

// 118. Pascal's Triangle
// List<int> x = Enumerable.Repeat(value, count).ToList();  
public class Solution {
    public IList<IList<int>> Generate(int numRows) {
        IList<IList<int>> result = new List<IList<int>>();        
        for( int i = 0; i < numRows; i++ )
        {
            IList<int> row = Enumerable.Repeat(1, i+1).ToList();
            for( int j = 1; j < i; j++ )
            {
                row[j] = result[i - 1][j] + result[i - 1][j - 1];
                
            }
            result.Add(row);
        }
        return result;
    }
}

// Day 5 Array
// 36. Valid Sudoku
// dict[number].Contains
// dict[number].Add
public class Solution {
    public bool IsValidSudoku(char[][] board) {
        Dictionary<char, List<string>> dict =
    new Dictionary<char, List<string>>();
    
    for (int i=0; i<9; ++i) {
        for (int j=0; j<9; ++j) {
            Char number = board[i][j];
            if (number != '.'){
                
                if(!dict.ContainsKey(number))
                {
                    dict[number] = new List<string>();
                }
                if (dict[number].Contains(number + " in row " + i) ||dict[number].Contains(number + " in column " + j) || dict[number].Contains(number + " in block " + i/3 + "-" + j/3)){
                    return false;
                }
                dict[number].Add(number + " in row " + i);

                dict[number].Add(number + " in column " + j);

                dict[number].Add(number + " in block " + i/3 + "-" + j/3);
                                
            }
                
        }
    }
    return true;
    }
}
// 74. Search a 2D Matrix
// Binary Search
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        if(matrix.Length ==0) return false;
        int l= 0, r= matrix.Length * matrix[0].Length;
        while(l<r)
        {
            int m = l+(r-l)/2;
            if(matrix[m/matrix[0].Length][m%matrix[0].Length]==target)
            {
                return true;
            }
            else if(matrix[m/matrix[0].Length][m%matrix[0].Length]> target)
            {
                r=m;
            }
            else
            {
                l = l+1;
            }
        }
        return false;
    }
}

// Day 6 String
// 387. First Unique Character in a String
// Complexity Analysis
// Time complexity : O(N) since we go through the string of length N two times.
// Space complexity : O(1) because English alphabet contains 26 letters.
public class Solution {
    public int FirstUniqChar(string s) {
        Dictionary<char,int> count = new Dictionary<char, int>();
        // build hash map : character and how often it appears
        for(int i =0; i<s.Length; i++)
        {
            char c = s[i];
            if (count.ContainsKey(c)) count[c] +=1;
            else count[c] = 1;
        }
        // find the index
        for(int i = 0; i<s.Length; i++){
            if(count[s[i]] ==1) return i;
        }
        return -1;
    }
}
// 383. Ransom Note
public class Solution {
    public bool CanConstruct(string ransomNote, string magazine) {
        Dictionary<Char, int> count = new Dictionary<Char, int>();
        for(int i =0; i<magazine.Length; i++) {
            if (count.ContainsKey(magazine[i])) count[magazine[i]] +=1;
            else count[magazine[i]] = 1;
        }
        for (int i =0; i<ransomNote.Length; i++) {
            if (count.ContainsKey(ransomNote[i])) count[ransomNote[i]] -=1;
            else return false;
            if (count[ransomNote[i]] < 0) return false;
        }
        return true;
    }
}

// 242. Valid Anagram
public class Solution {
    public bool IsAnagram(string s, string t) {
        int[] alphabet = new int[26];
        for (int i = 0; i < s.Length; i++) alphabet[s[i] - 'a']++;
        for (int i = 0; i < t.Length; i++) alphabet[t[i] - 'a']--;
        for (int i = 0; i < alphabet.Length; i++) if (alphabet[i] != 0) return false;
        return true;
    }
}

// Follow up: 
// What if the inputs contain Unicode characters? 
// How would you adapt your solution to such a case?
public class Solution {
    public bool IsAnagram(string s, string t) {
        Dictionary<Char,int> map = new Dictionary<Char,int>();
        if(s.Length!=t.Length){return false;}
        for(int i=0;i<s.Length;i++){
            if (map.ContainsKey(s[i])) map[s[i]]+=1;
            else map[s[i]]=1;
            if (map.ContainsKey(t[i])) map[t[i]]-=1;
            else map[t[i]]=-1;
            
        }
        foreach(int i in map.Values) if(i != 0) return false;
        return true;
    }
}

public class Solution {
    public bool IsAnagram(string s, string t) {
        return String.Concat(s.OrderBy(c => c)) == String.Concat(t.OrderBy(c => c));
    }
}

// Day 7 Linked List
// 141. Linked List Cycle
// Solution2: Fast + Slow pointers
// Time complexity: O(n) 
// Space complexity: O(1)

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public bool HasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null) {
          if (fast.next == null) return false;
          fast = fast.next.next;
          slow = slow.next;
          if (fast == slow) return true;
        }
        return false;
    }
}

// 21. Merge Two Sorted Lists
// Solution 2: priority_queue / mergesort
// Recursive O(n)
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
    public ListNode MergeTwoLists(ListNode list1, ListNode list2) {
        // If one of the list is emptry, return the other one.
        if(list2 == null) return list1 ;
        if(list1 == null) return list2 ;
        // The smaller one becomes the head.
        if(list1.val < list2.val) {
            list1.next = MergeTwoLists(list1.next, list2);
            return list1;
        } else {
            list2.next = MergeTwoLists(list1, list2.next);
            return list2;
        }
    }
}

//Solution 1: Iterative O(n)
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
    public ListNode MergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(0,null);
        ListNode tail= dummy;
        while(list1 != null && list2 != null) {
            if(list1.val < list2.val) {
                tail.next=list1;
                list1=list1.next;
            }else{
                tail.next=list2;
                list2=list2.next;
            }
            tail=tail.next;
        }
        
        if(list1!=null) tail.next = list1;
        if(list2!=null) tail.next = list2;
        
        return dummy.next;
        
    }
}

// 203. Remove Linked List Elements
//  Approach 3: Recursive Solution
// Time Complexity: O(N) --> Each Node in the list is visited once.
// Space Complexity: O(N) --> Recursion Stack space
// Where, N = Length of the input list.
//When the input node is an empty node, then there is nothing to delete,
// so we just return a null node back. (That's the first line)
//When the head of the input node is the target we want to delete,
// we just return head.next instead of head to skip it.
// (That's the third line), else we will return head.
//We apply the same thing to every other node until it reaches null. (That's the second line).
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
    public ListNode RemoveElements(ListNode head, int val) {
        if (head == null) return null;
        head.next = RemoveElements(head.next, val);
        return head.val == val ? head.next : head;
    }
}

public class Solution {
    public ListNode RemoveElements(ListNode head, int val) {
        if(head == null) return null;
        head.next = RemoveElements(head.next, val);
        if(head.val == val) return head.next;
        return head;
    }
}

// Approach 2: Iterative Solution using a Previous Pointer
// Time Complexity: O(N) --> Each Node in the list is visited once.
// Space Complexity: O(1) --> Contant space is used for this solution
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
    public ListNode RemoveElements(ListNode head, int val) {
        ListNode dummy = new ListNode(-1, head), prev = dummy;
        while( head != null){
            if(head.val != val) prev = head;   
            else prev.next = head.next;       
            head = head.next;
        } 
            
        return dummy.next;     
    }
}

// Approach 1: Iterative Solution without using a Previous Pointer
// Time Complexity: O(N) --> Each Node in the list is visited once.
// Space Complexity: O(1) --> Contant space is used for this solution
// Where, N = Length of the input list.
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
    public ListNode RemoveElements(ListNode head, int val) {
        ListNode dummy = new ListNode(0, null);
        dummy.next = head;
        ListNode curr = dummy;
        
        while(curr.next != null) {
            if(curr.next.val == val) {
                curr.next = curr.next.next;
            } else {
                curr = curr.next;
            }
        }
        
        return dummy.next;
    }
}

// Day 8 Linked List
// 206. Reverse Linked List
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
    public ListNode ReverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        ListNode next;
        while (curr != null) {
          next = curr.next;
          curr.next = prev;
          prev = curr;
          curr = next;
        }
        return prev;
    }
}

// 83. Remove Duplicates from Sorted List
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
        if(head == null || head.next == null)return head;
        head.next = DeleteDuplicates(head.next);
        return head.val == head.next.val ? head.next : head;
    }
}

public class Solution {
    public ListNode DeleteDuplicates(ListNode head) {
        if (head == null) return head;
        head.next = DeleteDuplicates(head.next);
        return head.next != null && head.val == head.next.val ? head.next : head;
    }
}

public class Solution {
    public ListNode DeleteDuplicates(ListNode head) {
        ListNode cur=head;
        while (cur!=null){
            while (cur.next!=null && cur.val==cur.next.val)                                 cur.next=cur.next.next;
            cur=cur.next;
        }
        return head;
    }
}

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
        ListNode current = head;
        while (current != null && current.next != null) {
            if (current.val == current.next.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return head;
    }
}

public class Solution {
    public ListNode DeleteDuplicates(ListNode head) {
        ListNode cur = head;
        while (cur != null) {
            while (cur.next!=null && cur.val == cur.next.val)
                cur.next = cur.next.next;
            cur = cur.next;
        }
        return head;
    }
}
// Day 9 Stack / Queue
// 20. Valid Parentheses
public class Solution {
    public bool IsValid(string s) {
        Stack<char> stack = new Stack<char>();
        foreach (char c in s.ToCharArray()) {
            if (c == '(')
                stack.Push(')');
            else if (c == '{')
                stack.Push('}');
            else if (c == '[')
                stack.Push(']');
            else if (stack.Count ==0 || stack.Pop() != c)
                return false;
        }
        return stack.Count==0;
    }
}

// 232. Implement Queue using Stacks
public class MyQueue {

    public Stack<int> s1 = new Stack<int>();
    public Stack<int> s2 = new Stack<int>();
    public int front;
    
    public MyQueue() {
        
    }
    
    public void Push(int x) {
        if (s1.Count == 0) front = x;
        s1.Push(x);
    }
    
    public int Pop() {
        if(s2.Count == 0)
        {
            while(s1.Count > 0)
            {
                s2.Push(s1.Pop());
            }
        }
        return s2.Pop();
    }
    
    public int Peek() {
        if(s2.Count > 0)
        {
            return s2.Peek();
        }
        return front;
    }
    
    public bool Empty() {
        return s1.Count == 0 && s2.Count == 0;
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.Push(x);
 * int param_2 = obj.Pop();
 * int param_3 = obj.Peek();
 * bool param_4 = obj.Empty();
 */

// Do you know when we should use two stacks to implement a queue?
// When there's only one thread doing the read/write operation to the stack, 
// there will always one stack empty. However, in a multi-thread application,
// if we have only one queue, for thread-safety, either read or write will lock the whole queue.
// In the two stack implementation, as long as the second stack is not empty,
// push operation will not lock the stack for pop.

/*
It's a common way to implement a queue in functional programming languages 
with purely functional (immutable, but sharing structure) lists (e.g. Clojure, Haskell, Erlang...):

use a pair of lists to represent a queue where elements are in FIFO order in the first list 
and in LIFO order in the second list

enqueue to the queue by prepending to the second list
dequeue from the queue by taking the first element of the first list

if the first list is empty: reverse the second list and replace the first list with it, 
and replace the second list with an empty list
(all operations return the new queue object in addition to any possible return values)

The point is that adding (removing) an element to (from) the front of a purely functional list is O(1) 
and the reverse operation which is O(n) is amortised over all the dequeues, 
so it's close to O(1), thereby giving you a ~O(1) queue implementation with immutable data structures.

This approach may be used to build a lock-free queue using two atomic single-linked list based stacks,
such as provided by Win32: Interlocked Singly Linked Lists.
The algorithm could be as described in liwp's answer, though the repacking step (bullet 4) can be optimized a bit.

Lock-free data structures and algorithms is a very exciting (to some of us) area of programming, 
but they must be used very carefully. In a general situation, lock-based algorithms are more efficient.
*/

// Day 10 Tree
// 144. Binary Tree Preorder Traversal

// Solution 1: Recursion
// Time complexity: O(n)
// Space complexity: O(n)

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
    public IList<int> PreorderTraversal(TreeNode root) {
        List<int> ans = new List<int>();    
        void preorder(TreeNode n) {
            if (n == null) return;
            ans.Add(n.val);
            preorder(n.left);
            preorder(n.right);
        };
        preorder(root);
        return ans;
    }
}

// Solution 2: Stack
// Time complexity: O(n)
// Space complexity: O(n)
public class Solution {
    public IList<int> PreorderTraversal(TreeNode root) {
        List<int> ans = new List<int>();
        Stack<TreeNode> s = new Stack<TreeNode>();
        if (root != null) s.Push(root);
        while (s.Count !=0) {
          TreeNode n = s.Peek();
          ans.Add(n.val);
          s.Pop();
          if (n.right != null) s.Push(n.right);
          if (n.left != null) s.Push(n.left);            
        }
        return ans;
    }
}

// 94. Binary Tree Inorder Traversal

// Solution: Recursion
// Time complexity: O(n)
// Space complexity: O(h)

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
    public IList<int> InorderTraversal(TreeNode root) {
        List<int> ans = new List<int>();
        inorderTraversal(root, ans);
        return ans;
      }
    
    public void inorderTraversal(TreeNode root, List<int> ans) {
        if (root == null) return;
        inorderTraversal(root.left, ans);
        ans.Add(root.val);
        inorderTraversal(root.right, ans); 
    }
}

// Solution 2: Iterative
// Time complexity: O(n)
// Space complexity: O(h)

public class Solution {
    public IList<int> InorderTraversal(TreeNode root) {
        if (root == null) return new List<int>();
        List<int> ans = new List<int>();
        Stack<TreeNode> s = new Stack<TreeNode>();
        TreeNode curr = root;
        while (curr != null || s.Count!=0) {
            while (curr != null) {
                s.Push(curr);
                curr = curr.left;
            }
            curr = s.Pop();
            ans.Add(curr.val);
            curr = curr.right;
        }    
        return ans;
    }
}

// Solution 3: Morris Traversal
// Time complexity: O(n)
// Space complexity: O(1)
/*
Step 1: Initialize current as root
Step 2: While current is not NULL,

If current does not have left child
    a. Add current’s value
    b. Go to the right, i.e., current = current.right

Else
    a. In current's left subtree, make current the right child of the rightmost node
    b. Go to this left child, i.e., current = current.left
*/
public class Solution {
    public IList<int> InorderTraversal(TreeNode root) {
        List<int> ans = new List<int>();
        TreeNode curr = root;
        TreeNode pre = new TreeNode();
        while (curr != null) {
            if (curr.left == null) {
                ans.Add(curr.val);
                curr = curr.right;
            }
            else{
                pre = curr.left;
                while (pre.right != null){
                    pre = pre.right;
                }
            
            pre.right = curr;
            TreeNode temp = curr;
            curr = curr.left;
            temp.left = null;
            }
        }    
        return ans;
    }
}
// 145. Binary Tree Postorder Traversal

// Solution: Recursion
// Time complexity: O(n)
// Space complexity: O(h)

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
    
    public IList<int> PostorderTraversal(TreeNode root) {
        List<int> ans = new List<int>();
        postorderTraversal(root, ans);
        return ans;
      }
    
    public void postorderTraversal(TreeNode root, List<int> ans) {
        if (root == null) return;
        postorderTraversal(root.left, ans);        
        postorderTraversal(root.right, ans); 
        ans.Add(root.val);
    }
}

// Solution 2: Iterative
// Time complexity: O(n)
// Space complexity: O(h)

public class Solution {
    
    public IList<int> PostorderTraversal(TreeNode root) {
        if (root == null) return new List<int>();
        List<int> ans = new List<int>();
        Stack<TreeNode> s = new Stack<TreeNode>();
        s.Push(root);
        while (s.Count!=0 ) {
            TreeNode n = s.Peek();
            s.Pop();
            ans.Insert(0, n.val);// O(1)
            if (n.left != null) s.Push(n.left);
            if (n.right != null) s.Push(n.right);
        }   

        return ans;
        
    }
}

public class Solution {
    
    public IList<int> PostorderTraversal(TreeNode root) {
        if (root == null) return new List<int>();
        List<int> ans = new List<int>();
        IList<int> l = PostorderTraversal(root.left);
        IList<int> r = PostorderTraversal(root.right);
        
        ans.AddRange(l);
        ans.AddRange(r);
        ans.Add(root.val);
        return ans;
    }
}

// Day 11 Tree
// 102. Binary Tree Level Order Traversal
// Solution 1: BFS O(n)
public class Solution {
    public IList<IList<int>> LevelOrder(TreeNode root) {       
        if(root==null) return new List<IList<int>>();;
        List<IList<int>> ans = new List<IList<int>>();
        List<TreeNode> curr = new List<TreeNode>(),next = new List<TreeNode>();
        curr.Add(root);
        while(curr.Count != 0) {
            ans.Add(new List<int>());
            foreach(TreeNode node in curr) {
                ans[ans.Count-1].Add(node.val);
                if(node.left!=null) next.Add(node.left);
                if(node.right!=null) next.Add(node.right);
            }
            List<TreeNode> temp = new List<TreeNode>();
            temp = next;
            next = curr;
            curr = temp;
            next.Clear();
        }
        return ans;
    }
}
// Solution 2: DFS O(n)
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
    public IList<IList<int>> LevelOrder(TreeNode root) {
        List<IList<int>> ans = new List<IList<int>>();
        DFS(root, 0 /* depth */, ans);
        return ans;
    }
    public void DFS(TreeNode root, int depth, List<IList<int>> ans) {
        if(root==null) return;
        // Works with pre/in/post order
        while(ans.Count<=depth){
            ans.Add(new List<int>());
        } 
        ans[depth].Add(root.val); // pre-order
        DFS(root.left, depth+1, ans);        
        DFS(root.right, depth+1, ans);  
    }
}

// 104. Maximum Depth of Binary Tree

//Solution: Recursion
//maxDepth(root) = max(maxDepth(root.left), maxDepth(root.right)) + 1
//Time complexity: O(n)
//Space complexity: O(n)

public class Solution {
    public int MaxDepth(TreeNode root) {
        if (root == null) return 0;
        int l = MaxDepth(root.left);
        int r = MaxDepth(root.right);
        return Math.Max(l, r) + 1;
    }
}

// 101. Symmetric Tree
//recursive solutions
public class Solution {
    public bool IsSymmetric(TreeNode root) {
        if (root == null ) return true;
        return helper(root.left, root.right);
    }
    
    public bool helper(TreeNode p, TreeNode q) {
        if (p ==null && q==null) {
            return true;
        } else if (p==null || q==null) {
            return false;
        }
        
        return p.val == q.val && helper(p.left,q.right) && helper(p.right, q.left); 
    }
}

// iterative solutions

public class Solution {
    public bool IsSymmetric(TreeNode root) {
        if (root == null) return true;
        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.Push(root.left);
        stack.Push(root.right);
        while (stack.Count != 0) {
            TreeNode n1 = stack.Pop(), n2 = stack.Pop();
            if (n1 == null && n2 == null) continue;
            if (n1 == null || n2 == null || n1.val != n2.val) return false;
            stack.Push(n1.left);
            stack.Push(n2.right);
            stack.Push(n1.right);
            stack.Push(n2.left);
        }
        return true;
    }
}


// Day 12 Tree
// 226. Invert Binary Tree

// recursively

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
    public TreeNode InvertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        TreeNode left = root.left, right = root.right;
        root.left = InvertTree(right);
        root.right = InvertTree(left);
        return root;
    }
}

// DFS
public class Solution {
    public TreeNode InvertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        Stack<TreeNode> stack = new Stack<TreeNode>();
        stack.Push(root);
        
        while(stack.Count != 0) {
            TreeNode node = stack.Pop();
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;
            
            if(node.left != null) {
                stack.Push(node.left);
            }
            if(node.right != null) {
                stack.Push(node.right);
            }
        }
        return root;
    }
}

// BFS
public class Solution {
    public TreeNode InvertTree(TreeNode root) {
        if (root == null) {
            return null;
        }

        Queue<TreeNode> queue = new Queue<TreeNode>();
        queue.Enqueue(root);

        while(queue.Count != 0) {
            TreeNode node = queue.Dequeue();
            TreeNode left = node.left;
            node.left = node.right;
            node.right = left;

            if(node.left != null) {
                queue.Enqueue(node.left);
            }
            if(node.right != null) {
                queue.Enqueue(node.right);
            }
        }
        return root;
    }
}

// 112. Path Sum 

// Solution: Recursion
// Time complexity: O(n)
// Space complexity: O(n)

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
    public bool HasPathSum(TreeNode root, int targetSum) {
        if (root==null) return false;
        if (root.left ==null && root.right ==null ) return root.val==targetSum;
        int new_sum = targetSum - root.val;
        return HasPathSum(root.left, new_sum) || HasPathSum(root.right, new_sum);
      
    }
}
// Day 13 Tree
// 700. Search in a Binary Search Tree

// Solution: Recursion
// Time complexity: O(logn ~ n)
// Space complexity: O(logn ~ n)

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
    public TreeNode SearchBST(TreeNode root, int val) {
        if (root == null) return null;
        if (val == root.val) return root;
        else if (val > root.val) return SearchBST(root.right, val);
        return SearchBST(root.left, val);
    }
}


// 701. Insert into a Binary Search Tree

// Solution: Recursion
// Time complexity: O(logn ~ n)
// Space complexity: O(logn ~ n)

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
    public TreeNode InsertIntoBST(TreeNode root, int val) {
        if (root == null) return new TreeNode(val);
        if (val > root.val)
          root.right = InsertIntoBST(root.right, val);
        else
          root.left = InsertIntoBST(root.left, val);
        return root;
    }
}

// Day 14 Tree
// 98. Validate Binary Search Tree

// Solution 1
// Traverse the tree and limit the range of each subtree and check whether root’s value is in the range.
// Time complexity: O(n)
// Space complexity: O(n)
// Note: in order to cover the range of -2^31 ~ 2^31-1, we need to use long or nullable integer.

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
    public bool IsValidBST(TreeNode root) {
        return isValidBST(root, null, null);
  }
  
    public bool isValidBST(TreeNode root, int? min, int? max) {
        if (root == null) return true;
        if ((min != null && root.val <= min) 
          ||(max != null && root.val >= max))
            return false;

        return isValidBST(root.left, min, root.val)
            && isValidBST(root.right, root.val, max);
    }
}

public class Solution {
    public bool IsValidBST(TreeNode root) {
        return isValidBST(root, long.MinValue, long.MaxValue);
  }
  
    public bool isValidBST(TreeNode root, long min, long max) {
        if (root == null) return true;
        if ((min != null && root.val <= min) 
          ||(max != null && root.val >= max))
            return false;

        return isValidBST(root.left, min, root.val)
            && isValidBST(root.right, root.val, max);
    }
}

// Solution 2
// Do an in-order traversal, the numbers should be sorted,
// thus we only need to compare with the previous number.
// Time complexity: O(n)
// Space complexity: O(n)

public class Solution {
    public TreeNode prev;
    public bool IsValidBST(TreeNode root) {
        prev = null;
        return inOrder(root);
  }
  
    public bool inOrder(TreeNode root) {
        if (root == null) return true;
        if (!inOrder(root.left)) return false;
        if (prev != null && root.val <= prev.val) return false;
        prev = root;
        return inOrder(root.right);
    }
}



// 653. Two Sum IV - Input is a BST

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
    List<int> s = new List<int>();
    public bool FindTarget(TreeNode root, int k) {
        if (root == null) return false;
        if (s.Contains(k - root.val) ) return true;
        s.Add(root.val);
        return FindTarget(root.left, k) || FindTarget(root.right, k);
    }
}



// 235. Lowest Common Ancestor of a Binary Search Tree

// Solution: Recursion
// Time complexity: O(n)
// Space complexity: O(n)

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int x) { val = x; }
 * }
 */

public class Solution {
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val) 
            return LowestCommonAncestor(root.left, p, q);
        if (p.val > root.val && q.val > root.val)
            return LowestCommonAncestor(root.right, p, q);
        return root;
    }
}
// Day 1 Array
// 136. Single Number
/*
we use bitwise XOR to solve this problem :
first , we have to know the bitwise XOR in C//
0 ^ N = N
N ^ N = 0

It's worth noting that this XOR solution is not a generic "find the non duplicate" function. 
It works only as long as you know each other number appears exactly twice 
(which is indeed what the question states, so it's totally valid here). 
But like, if the array was allowed to contain an arbitrary number of duplicates 
(ie. the same number appears 3 or 5 times), then this solution breaks down. 
This solution works if the duplicates always appear an even number of times (2x, 4x, 6x, etc.). 
And the time complexity is still O(N). 
Whereas a simple object/count solution is also O(N), and solves any number of duplicates. 
The big win with this solution is memory. You don't need to keep track of anything, which is cool.
*/

public class Solution {
    public int SingleNumber(int[] nums) {
        int result = nums[0];
        for(int i=1; i<nums.Length; i++)
            result ^= nums[i]; /* Get the xor of all elements */

        return result;

    }
}


// 169. Majority Element
// Approach 3: Sorting
// Time complexity : O(nlgn)O(nlgn)
// Space complexity : O(1) or O(n)
public class Solution {
    public int MajorityElement(int[] nums) {
        Array.Sort(nums);
        return nums[nums.Length/2];
    }
}
// Approach 5: Divide and Conquer
// Time complexity : O(nlgn)
// Space complexity : O(lgn)
public class Solution {
    public int MajorityElement(int[] nums) {
        return majorityElementRec(nums, 0, nums.Length-1);
    }
    
    private int countInRange(int[] nums, int num, int l, int r) {
        int count = 0;
        for (int i = l; i <= r; i++) {
            if (nums[i] == num) {
                count++;
            }
        }
        return count;
    }
    
    public int majorityElementRec(int[] nums, int l, int r) {
        // base case; the only element in an array of size 1 is the majority
        // element.
        if (l == r) {
            return nums[l];
        }

        // recurse on left and right halves of this slice.
        int m = l+ (r-l)/2 ;
        int ml = majorityElementRec(nums, l, m);
        int mr = majorityElementRec(nums, m+1, r);

        // if the two halves agree on the majority element, return it.
        if (ml == mr) {
            return ml;
        }

        // otherwise, count each element and return the "winner".
        int leftCount = countInRange(nums, ml, l, r);
        int rightCount = countInRange(nums, mr, l, r);

        return leftCount > rightCount ? ml : mr;
    }
}

// 15. 3Sum

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

// Day 2 Array
// 75. Sort Colors
public class Solution {
    public void SortColors(int[] nums) {
        int zero = 0; 
        int r = nums.Length - 1; 
        int l = 0; 
        
        while(l <= r) {
            if(nums[l] == 0) {
                swap(nums, zero++, l++);
            } 
            else if(nums[l] == 2) {
                swap(nums, l, r--);
            } 
            else {
                l++;
            }
        }
    }
    
    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }
}

// 56. Merge Intervals
// Approach 2: Sorting
// Time complexity : O(nlogn)
// Space complexity : O(logN) or O(n)
public class Solution {
    public int[][] Merge(int[][] intervals) {
        //Array.Sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        //intervals.OrderBy(r => r[0])
        List<int[]> merged = new List<int[]>();
        foreach (int[] interval in intervals.OrderBy(r => r[0])) {
            // if the list of merged intervals is empty or if the current
            // interval does not overlap with the previous, simply append it.
            if (merged.Count==0 || merged[merged.Count-1][1] < interval[0]) {
                merged.Add(interval);
            }
            // otherwise, there is overlap, so we merge the current and previous
            // intervals.
            else {
                merged[merged.Count-1][1] = Math.Max(merged[merged.Count-1][1], interval[1]);
            }
        }
        return merged.ToArray();
    }
}

// 706. Design HashMap

public class MyHashMap {

    int[] data;
    public MyHashMap() {
        data = new int[1000001];
        Array.Fill(data, -1);
    }
    
    public void Put(int key, int value) {
        data[key] = value;
    }
    
    public int Get(int key) {
         return data[key];
    }
    
    public void Remove(int key) {
        data[key] = -1;
    }
    
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap obj = new MyHashMap();
 * obj.Put(key,value);
 * int param_2 = obj.Get(key);
 * obj.Remove(key);
 */

// Day 3 Array
// 119. Pascal's Triangle II

public class Solution {
    public IList<int> GetRow(int rowIndex) {
        int[] ans = new int[rowIndex+1];
        ans[0] = 1;
        ans[rowIndex] = 1;
        long temp = 1;
        int up = rowIndex;
        for(int i=1; i < rowIndex; i++){
            temp = temp * up / i;
            ans[i]=(int)temp;
            up--;
        }
        return ans;
    }
}

// 48. Rotate Image

public class Solution {
    public void Rotate(int[][] matrix) {
        int n = matrix.Length;
        for (int i = 0; i < (n + 1) / 2; i ++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1];
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 -i];
                matrix[j][n - 1 - i] = matrix[i][j];
                matrix[i][j] = temp;
            }
        }
    }
}

// 59. Spiral Matrix II
public class Solution {
    public int[][] GenerateMatrix(int n) {
        int[][] result = new int[n][];
        int cnt = 1;
        for (int i = 0; i < n; i++) {
            result[i] = new int[n];
        }
        for (int layer = 0; layer < (n + 1) / 2; layer++) {
            
            // direction 1 - traverse from left to right
            for (int ptr = layer; ptr < n - layer; ptr++) {
                result[layer][ptr] = cnt++;
            }
            // direction 2 - traverse from top to bottom
            for (int ptr = layer + 1; ptr < n - layer; ptr++) {
                result[ptr][n - layer - 1] = cnt++;
            }
            // direction 3 - traverse from right to left
            for (int ptr = layer + 1; ptr < n - layer; ptr++) {
                result[n - layer - 1][n - ptr - 1] = cnt++;
            }
            // direction 4 - traverse from bottom to top
            for (int ptr = layer + 1; ptr < n - layer - 1; ptr++) {
                result[n - ptr - 1][layer] = cnt++;
            }
        }
        return result;
    }
}


public class Solution {
    public int[][] GenerateMatrix(int n) {
    int[][] ret = new int[n][];
	int left = 0,top = 0;
	int right = n -1,down = n - 1;
	int count = 1;
    for (int i = 0; i < n; i++) {
        ret[i] = new int[n];
    }
	while (left <= right) {
		for (int j = left; j <= right; j ++) {
			ret[top][j] = count++;
		}
		top ++;
		for (int i = top; i <= down; i ++) {
			ret[i][right] = count ++;
		}
		right --;
		for (int j = right; j >= left; j --) {
			ret[down][j] = count ++;
		}
		down --;
		for (int i = down; i >= top; i --) {
			ret[i][left] = count ++;
		}
		left ++;
	}
	return ret;
    }
}
// Day 4 Array
// 240. Search a 2D Matrix II
public class Solution {
    public bool SearchMatrix(int[][] matrix, int target) {
        if(matrix == null || matrix.Length < 1 || matrix[0].Length <1) {
            return false;
        }
        int col = matrix[0].Length-1;
        int row = 0;
        while(col >= 0 && row <= matrix.Length-1) {
            if(target == matrix[row][col]) {
                return true;
            } else if(target < matrix[row][col]) {
                col--;
            } else if(target > matrix[row][col]) {
                row++;
            }
        }
        return false;
    }
}

// 435. Non-overlapping Intervals

public class Solution {
    public int EraseOverlapIntervals(int[][] intervals) {
        if (intervals.Length == 0) return 0;
        //intervals[1].OrderBy(x => x).ToArray();
        //intervals.OrderBy(x => x[1]).ToArray();
        Array.Sort(intervals, (x, y) => x[1].CompareTo(y[1]));
        
        int end = intervals[0][1];
        int count = 0;
        for (int i = 1; i < intervals.Length; i++) {
            if (intervals[i][0] >= end) end = intervals[i][1];
            else count++;
        }
        return count;
    }
}
// Day 5 Array
// 334. Increasing Triplet Subsequence
public class Solution {
    public bool IncreasingTriplet(int[] nums) {
        // start with two largest values, as soon as we find a number bigger than both, while both have been updated, return true.
        int small =int.MaxValue, big = int.MaxValue;
        foreach (int n in nums) {
            if (n <= small) { small = n; } // update small if n is smaller than both
            else if (n <= big) { big = n; } // update big only if greater than small but smaller than big
            else return true; // return if you find a number bigger than both
        }
        return false;
    }
}

// 238. Product of Array Except Self
/*
Suppose you have numbers:
Numbers [1      2       3       4       5]
Pass 1: [1  ->  1  ->   12  ->  123  -> 1234]
Pass 2: [2345 <-345 <-  45  <-  5   <-  1]

Finally, you multiply ith element of both the lists to get:
Pass 3: [2345, 1345, 1245, 1235, 1234]
*/
public class Solution {
    public int[] ProductExceptSelf(int[] nums) {
        int[] result = new int[nums.Length];
        int tmp = 1;
        for (int i = 0; i < nums.Length; i++) {
            result[i] = tmp;
            tmp *= nums[i];
        }
        tmp = 1;
        for (int i = nums.Length - 1; i >= 0; i--) {
            result[i] *= tmp;
            tmp *= nums[i];
        }
        return result;
    }
}

// 560. Subarray Sum Equals K
public class Solution {
    public int SubarraySum(int[] nums, int k) {
        int sum = 0, result = 0;
        Dictionary<int, int> preSum = new Dictionary<int, int>();
        preSum.Add(0, 1);
        
        for (int i = 0; i < nums.Length; i++) {
            sum += nums[i];
            if (preSum.ContainsKey(sum - k)) {
                result += preSum[sum - k];
            }
            
            if (!preSum.ContainsKey(sum))
            {
                preSum.Add(sum, 1);
            }
            else{
                preSum[sum] += 1;
            }
            
        }
        
        return result;
    }
}
// Day 6 String
// 415. Add Strings

public class Solution {
    public string AddStrings(string num1, string num2) {
        StringBuilder sb = new StringBuilder();
        int carry = 0;
        for(int i = num1.Length - 1, j = num2.Length - 1; i >= 0 || j >= 0 || carry == 1; i--, j--){
            int x = i < 0 ? 0 : num1[i] - '0';
            int y = j < 0 ? 0 : num2[j] - '0';
            sb.Append((x + y + carry) % 10);
            carry = (x + y + carry) / 10;
        }
        char[] charArray = sb.ToString().ToCharArray();
        Array.Reverse(charArray);
        
        return new string(charArray);
    }
}

// 409. Longest Palindrome
/* Approach : Greedy [Accepted]

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
*/
public class Solution {
    public int LongestPalindrome(string s) {
        int[] count = new int[128];
        foreach (char c in s.ToCharArray())
            count[c]++;

        int ans = 0;
        foreach (int v in count) {
            ans += v / 2 * 2;
            if (ans % 2 == 0 && v % 2 == 1)
                ans++;
        }
        return ans;
    }
}

// Day 7 String

// 290. Word Pattern
public class Solution {
    public bool WordPattern(string pattern, string s) {
        String[] words = s.Split(" ");
        if (words.Length != pattern.Length) return false;
        Dictionary<char,string> table = new Dictionary<char,string>();

        for( int i = 0; i < words.Length; i++){
            
            if(table.ContainsKey(pattern[i])){
                if(table[pattern[i]]!=words[i])
                    return false;
            }else{
                if(table.ContainsValue(words[i]))
                    return false;
                table.Add(pattern[i], words[i]);
            }    
        }
        return true;
    }
}

// 763. Partition Labels
/* Approach : Greedy
Complexity Analysis
Time Complexity: O(N), where NN is the length of SS.
Space Complexity: O(1) to keep data structure last of not more than 26 characters.
*/
public class Solution {
    public IList<int> PartitionLabels(string s) {
        int[] last = new int[26];
        for (int i = 0; i < s.Length; ++i)
            last[s[i] - 'a'] = i;
        
        int j = 0, anchor = 0;
        List<int> ans = new List<int>();
        for (int i = 0; i < s.Length; ++i) {
            j = Math.Max(j, last[s[i] - 'a']);
            if (i == j) {
                ans.Add(i - anchor + 1);
                anchor = i + 1;
            }
        }
        return ans;
    }
}

// Day 8 String

// 49. Group Anagrams
// Use char[26] as bucket to count the frequency instead of Arrays.sort,
// this can reduce the O(nlgn) to O(n) when calculate the key.
public class Solution {
    public IList<IList<string>> GroupAnagrams(string[] strs) {
        if(strs == null || strs.Length == 0) return new List<IList<string>>();
        Dictionary<String, IList<String>> map = new Dictionary<String, IList<String>>();
        foreach(String s in strs){

            //char type 0~127 is enough for constraint 0 <= strs[i].length <= 100
            //char array to String is really fast, thanks @legendaryengineer
            //You should use other data type when length of string is longer.
            char[] frequencyArr = new char[26];   
            foreach(char c in s.ToCharArray()){
                frequencyArr[c - 'a']++;
            }
            String key = new String(frequencyArr);
            //char[] Convert.ToString() could not work! 
            //char[] ToString() could not work! 
            //char[] in new string() works!
            
            /* this part is same with below
            if (!map.ContainsKey(key)) map[key] = new List<String>();
            map[key].Add(s);
            */
            
            IList<String> tempList = map.GetValueOrDefault(key, new List<String>());
            tempList.Add(s);
            map[key] = tempList;
        }

        //return new List<IList<string>>(map.Values);   
        return map.Values.ToList();

    }
}

// 43. Multiply Strings
// Compute the ones-digit, then the tens-digit, then the hundreds-digit, etc. 
// For example when multiplying 1234 with 5678, the thousands-digit of the product is 4*5 + 3*6 + 2*7 + 1*8 (plus what got carried from the hundreds-digit).
public class Solution {
    public string Multiply(string num1, string num2) {
        if (String.Equals("0",num1) || String.Equals("0",num2))
			return "0";

		int[] ans = new int[num1.Length + num2.Length - 1];

		for (int i = 0; i < num1.Length; i++) {
			for (int j = 0; j < num2.Length; j++) {
				ans[i + j] += (num1[i] - '0') * (num2[j] - '0');
			}
		}

		for (int i = ans.Length - 1; i > 0; i--) {
			ans[i - 1] += ans[i] / 10;
			ans[i] %= 10;
		}

		StringBuilder sb = new StringBuilder();
		foreach (int i in ans) {
			sb.Append(i);
		}

		return sb.ToString();
    }
}

// Day 9 String

// 187. Repeated DNA Sequences
// Count occurences of all possible substring with length = 10, there are total 10*(N-9) substrings.
// Just return all substrings which occur more than once.
// Complexity
// Time & Space: O(10*N)
public class Solution {
    public IList<string> FindRepeatedDnaSequences(string s) {
         Dictionary<String, int> cnt = new Dictionary<String, int>();
        for (int i = 0; i < s.Length - 9; i++) {
            String subStr = s.Substring(i, 10);
            cnt[subStr] = cnt.GetValueOrDefault(subStr, 0) + 1;
        }
        List<String> ans = new List<String>();
        foreach (String key in cnt.Keys){
            if (cnt[key] >= 2){
                ans.Add(key);
            }       
        }
            
        return ans;

    }
}

// 5. Longest Palindromic Substring

// Day 10 Linked List

// 2. Add Two Numbers

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
    public ListNode AddTwoNumbers(ListNode l1, ListNode l2) {
        int carry = 0;
    ListNode p, dummy = new ListNode(0);
    p = dummy;
    while (l1 != null || l2 != null || carry != 0) {
        if (l1 != null) {
            carry += l1.val;
            l1 = l1.next;
        }
        if (l2 != null) {
            carry += l2.val;
            l2 = l2.next;
        }
        p.next = new ListNode(carry%10);
        carry /= 10;
        p = p.next;
    }
    return dummy.next;
    }
}

// Approach 1: Elementary Math
// Complexity Analysis
// Time complexity : O(max(m,n)). Assume that mm and nn represents the length of l1 and l2 respectively, 
// the algorithm above iterates at most max(m,n) times.
// Space complexity : O(max(m,n)). The length of the new list is at most max(m,n)+1.
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
    public ListNode AddTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummyHead = new ListNode(0);
        ListNode p = l1, q = l2, curr = dummyHead;
        int carry = 0;
        while (p != null || q != null) {
            int x = (p != null) ? p.val : 0;
            int y = (q != null) ? q.val : 0;
            int sum = carry + x + y;
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (p != null) p = p.next;
            if (q != null) q = q.next;
        }
        if (carry > 0) {
            curr.next = new ListNode(carry);
        }
        return dummyHead.next;
    }
}

// 142. Linked List Cycle II
/*Solution : Fast and Slow

We have 2 phases:
Phase 1: Use Fast and Slow to find the intersection point, 
if intersection point == null then there is no cycle.
Phase 2: Since F = b + m*C, where m >= 0 (see following picture), 
we move head and intersection as the same time, util they meet together, 
the meeting point is the cycle pos.
Complexity:

Time: O(N), where N <= 10^4 is number of elements in the linked list.
Space: O(1)*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode DetectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
    
        // Phase 1: Find the intersection node
        ListNode intersect = findIntersect(head);
        if( intersect == null) return null;
        
        // Phase 2: Find the cycle node
        while (head != intersect){
            head = head.next;
            intersect = intersect.next;
        }
            
        return head;
    }
    public ListNode findIntersect(ListNode head){
            ListNode slow = head;
            ListNode fast = head;
            while(fast != null && fast.next != null){
                slow = slow.next;
                fast = fast.next.next;
                if (slow == fast)
                    return slow;
            }     
            return null;
    }
            
}

// Day 11 Linked List

// 160. Intersection of Two Linked Lists
/*
I found most solutions here preprocess linkedlists to get the difference in len.
Actually we don't care about the "value" of difference, we just want to make sure two pointers reach the intersection node at the same time.

We can use two iterations to do that. 
In the first iteration, we will reset the pointer of one linkedlist to the head of another linkedlist after it reaches the tail node. 
In the second iteration, we will move two pointers until they points to the same node. 
Our operations in first iteration will help us counteract the difference. 
So if two linkedlist intersects, the meeting point in second iteration must be the intersection point. 
If the two linked lists have no intersection at all, then the meeting pointer in second iteration must be the tail node of both lists, which is null
*/
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     public int val;
 *     public ListNode next;
 *     public ListNode(int x) { val = x; }
 * }
 */
public class Solution {
    public ListNode GetIntersectionNode(ListNode headA, ListNode headB) {
        //boundary check
    if(headA == null || headB == null) return null;
    
    ListNode a = headA;
    ListNode b = headB;
    
    //if a & b have different len, then we will stop the loop after second iteration
    while( a != b){
    	//for the end of first iteration, we just reset the pointer to the head of another linkedlist
        a = a == null? headB : a.next;
        b = b == null? headA : b.next;    
    }
    
    return a;
    }
}

// 82. Remove Duplicates from Sorted List II

// Day 12 Linked List

// 24. Swap Nodes in Pairs

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
    public ListNode SwapPairs(ListNode head) {
        if (head == null || head.next == null) return head;    
 
        ListNode d = new ListNode(0);
        d.next = head;
        head = d;

        while (head !=null && head.next!=null && head.next.next!=null) {
          ListNode n1 = head.next;
          ListNode n2 = n1.next;

          n1.next = n2.next;
          n2.next = n1;

          head.next = n2;
          head = n1;
        }
        return d.next;
    }
}

// 707. Design Linked List

public class MyLinkedList {

    class Node {
    public int val;
    public Node next;
    public Node(int val) { this.val = val; this.next = null; }
    public Node(int val, Node next) { this.val = val; this.next = next; }
  }
  
    private Node head;
    private Node tail;
    private int size;
    
    public MyLinkedList() {
        this.head = this.tail = null;
        this.size = 0;
    }
    
    private Node getNode(int index) {
        Node n = new Node(0, this.head);
        while (index-- >= 0) {
          n = n.next;
        }
        return n;
    }
    
    public int Get(int index) {
        if (index < 0 || index >= size) return -1;
        return getNode(index).val;
    }
    
    public void AddAtHead(int val) {
        this.head = new Node(val, this.head);
        if (this.size++ == 0)
            this.tail = this.head;  
    }
    
    public void AddAtTail(int val) {
         Node n = new Node(val);
        if (this.size++ == 0)
          this.head = this.tail = n;
        else
          this.tail = this.tail.next = n;
    }
    
    public void AddAtIndex(int index, int val) {
        if (index < 0 || index > this.size) return;
        if (index == 0)  { this.AddAtHead(val); return; }
        if (index == size) { this.AddAtTail(val); return; }
        Node prev = this.getNode(index - 1);
        prev.next = new Node(val, prev.next);
        ++this.size;
    }
    
    public void DeleteAtIndex(int index) {
        if (index < 0 || index >= this.size) return;
        Node prev = this.getNode(index - 1);
        prev.next = prev.next.next;
        if (index == 0) this.head = prev.next;
        if (index == this.size - 1) this.tail = prev;
        --this.size;
    }
}


/**
 * Your MyLinkedList object will be instantiated and called as such:
 * MyLinkedList obj = new MyLinkedList();
 * int param_1 = obj.Get(index);
 * obj.AddAtHead(val);
 * obj.AddAtTail(val);
 * obj.AddAtIndex(index,val);
 * obj.DeleteAtIndex(index);
 */

// Day 13 Linked List

// 25. Reverse Nodes in k-Group

// recursive
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
    public ListNode ReverseKGroup(ListNode head, int k) {
        ListNode curr = head;
        int count = 0;
        while (curr != null && count != k) { // find the k+1 node
            curr = curr.next;
            count++;
        }
        if (count == k) { // if k+1 node is found
            curr = ReverseKGroup(curr, k); // reverse list with k+1 node as head
            // head - head-pointer to direct part, 
            // curr - head-pointer to reversed part;
            while (count > 0) { // reverse current k-group: 
                ListNode tmp = head.next; // tmp - next head in direct part
                head.next = curr; // preappending "direct" head to the reversed list 
                curr = head; // move head of reversed part to a new node
                head = tmp; // move "direct" head to the next node in direct part
                count--;
            }
            head = curr;
        }
        return head;
    }
}

// This type of question can split into 2 steps to solve ( like reverse in pair , reverse in 3 node ... )
// 1. test weather we have more then k node left, if less then k node left we just return head 
// 2. reverse k node at current level 
public class Solution {
    public ListNode ReverseKGroup(ListNode head, int k) {
       //1. test weather we have more then k node left, if less then k node left we just return head 
    ListNode node = head;
    int count = 0;
    while (count < k) { 
        if(node == null)return head;
        node = node.next;
        count++;
    }
    // 2.reverse k node at current level 
       ListNode pre = ReverseKGroup(node, k); //pre node point to the the answer of sub-problem 
        while (count > 0) {  
            ListNode next = head.next; 
            head.next = pre; 
            pre = head; 
            head = next;
            count = count - 1;
        }
        return pre;
    }
}

// Non-recursive
public class Solution {
    public ListNode ReverseKGroup(ListNode head, int k) {
       int n = 0;
        for (ListNode i = head; i != null; n++, i = i.next);
        
        ListNode dmy = new ListNode(0);
        dmy.next = head;
        for(ListNode prev = dmy, tail = head; n >= k; n -= k) {
            for (int i = 1; i < k; i++) {
                ListNode next = tail.next.next;
                tail.next.next = prev.next;
                prev.next = tail.next;
                tail.next = next;
            }
            
            prev = tail;
            tail = tail.next;
        }
        return dmy.next;
    }
}

// 143. Reorder List
// 3-step
// Time  Complexity: O(N)
// Space Complexity: O(1)
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
    public void ReorderList(ListNode head) {
        if(head==null||head.next==null) return;
            
            //step 1: find the middle pointer of the linked list and split the linked list into two halves using slow and fast pointers
            ListNode p1=head;
            ListNode p2=head;
            while(p2.next!=null&&p2.next.next!=null){ 
                p1=p1.next;
                p2=p2.next.next;
            }
            
            //step 2: reverse the last half linked list  1->2->3->4->5->6 to 1->2->3->6->5->4
            ListNode preMiddle=p1;
            ListNode preCurrent=p1.next;
            while(preCurrent.next!=null){
                ListNode current=preCurrent.next;
                preCurrent.next=current.next;
                current.next=preMiddle.next;
                preMiddle.next=current;
            }
            
            //Start reorder one by one  1->2->3->6->5->4 to 1->6->2->5->3->4
            p1=head;
            p2=preMiddle.next;
            while(p1!=preMiddle){
                preMiddle.next=p2.next;
                p2.next=p1.next;
                p1.next=p2;
                p1=p2.next;
                p2=preMiddle.next;
            }
    }
}
// Using List or Queue or Stack 
// Time  Complexity: O(N)
// Space Complexity: O(N)
public class Solution {
    public void ReorderList(ListNode head) {
        List<ListNode> dq = new List<ListNode>();
        ListNode p = head.next;
        while (p != null) {
            dq.Add(p);
            p = p.next;
        }
        p = head;
        while (dq.Count > 0) {
            p.next = dq[dq.Count - 1];
            p = p.next;
            dq.RemoveAt(dq.Count - 1);
            if (dq.Count > 0) {
                p.next = dq[0];
                p = p.next;
                dq.RemoveAt(0);
            }
        }
        p.next = null;
    }
}
// Day 14 Stack / Queue

// 155. Min Stack
public class MinStack {

    int min = int.MaxValue;
    Stack<int> stack;
    public MinStack() {
        stack = new Stack<int>();
    }
    
    public void Push(int val) {
        // only push the old minimum value when the current 
        // minimum value changes after pushing the new value x
        if(val <= min){          
            stack.Push(min);
            min=val;
        }
        stack.Push(val);
    }
    
    public void Pop() {
        // if pop operation could result in the changing of the current minimum value, 
        // pop twice and change the current minimum value to the last minimum value.
        if(stack.Pop() == min) min=stack.Pop();
    }
    
    public int Top() {
        return stack.Peek();
    }
    
    public int GetMin() {
        return min;
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.Push(val);
 * obj.Pop();
 * int param_3 = obj.Top();
 * int param_4 = obj.GetMin();
 */

// 1249. Minimum Remove to Make Valid Parentheses
/*
'''
Idea
Use stack to remove invalid mismatching parentheses, that is:
Currently, meet closing-parentheses but no opening-parenthesis in the previous -> remove current closing-parenthesis. For example: s = "())".
If there are redundant opening-parenthesis at the end, for example: s = "((()".

Complexity
Time: O(N), where N <= 10^5 is length of string s.
Space: O(N)
'''
*/
public class Solution {
    public string MinRemoveToMakeValid(string s) {
        Stack<int> openSt = new Stack<int>();
        List<int> todoRemove = new List<int>();
        for (int i = 0; i < s.Length; i++) {
            if (s[i] == '(')
                openSt.Push(i);
            else if (s[i] == ')') {
                if (openSt.Count==0)
                    todoRemove.Add(i); // Meet closing-parentheses but no opening-parenthesis -> remove closing-parenthesis 
                else
                    openSt.Pop();
            }
        }
        // remove remain opening-parenthesis
        while (openSt.Count != 0)
            todoRemove.Add(openSt.Pop());

        StringBuilder stringBuilder = new StringBuilder();
        for (int i = 0; i < s.Length; i++)
            if (!todoRemove.Contains(i))
                stringBuilder.Append(s[i]);

        return stringBuilder.ToString();
    }
}

// 1823. Find the Winner of the Circular Game

public class Solution {
    public int FindTheWinner(int n, int k) {
        int res = 0;
        for (int i = 1; i <= n; ++i)
            res = (res + k) % i;
        return res + 1;
    }
}

// Day 15 Tree

// 108. Convert Sorted Array to Binary Search Tree
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
    public TreeNode SortedArrayToBST(int[] nums) {
        return buildBST(nums, 0, nums.Length - 1);
  }
  
  public TreeNode buildBST(int[] nums, int l, int r) {
    if (l > r) return null;
    int m = l + (r - l) / 2;
    TreeNode root = new TreeNode(nums[m]);
    root.left = buildBST(nums, l, m - 1);
    root.right = buildBST(nums, m + 1, r);
    return root;
  }
    
}

// 105. Construct Binary Tree from Preorder and Inorder Traversal
/*
The basic idea is here:
Say we have 2 arrays, PRE and IN.
Preorder traversing implies that PRE[0] is the root node.
Then we can find this PRE[0] in IN, say it's IN[5].
Now we know that IN[5] is root, so we know that IN[0] - IN[4] is on the left side, 
IN[6] to the end is on the right side.
Recursively doing this on subarrays, we can build a tree out of it :)
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
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
         return helper(0, 0, inorder.Length - 1, preorder, inorder);
    }

    public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
        if (preStart > preorder.Length - 1 || inStart > inEnd) {
            return null;
        }
        TreeNode root = new TreeNode(preorder[preStart]);
        int inIndex = 0; // Index of current root in inorder
        for (int i = inStart; i <= inEnd; i++) {
            if (inorder[i] == root.val) {
                inIndex = i;
            }
        }
        root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
        root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
        return root;
    }
}
/*
The Root of the tree is the first element in Preorder Array.
Find the position of the Root in Inorder Array.
Elements to the left of Root element in Inorder Array are the left
subtree.
Elements to the right of Root element in Inorder Array are the right
subtree.
Call recursively buildTree on the subarray composed by the elements
in the left subtree. Attach returned left subtree root as left child
of Root node.
Call recursively buildTree on the subarray composed by the elements
in the right subtree. Attach returned right subtree root as right
child of Root node.
return Root.
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
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
         if(inorder.Length==0 || preorder.Length==0) return null;
        TreeNode root = new TreeNode(preorder[0]);
        if(preorder.Length==1) return root;
        int breakindex = -1;
        for(int i=0;i<inorder.Length;i++) { if(inorder[i]==preorder[0]) { breakindex=i; break;} }
        int[] subleftpre  = preorder.Skip(1).Take(breakindex+1).ToArray();
        int[] subleftin   = inorder.Skip(0).Take(breakindex).ToArray();
        int[] subrightpre = preorder.Skip(breakindex+1).Take(preorder.Length).ToArray();
        int[] subrightin  = inorder.Skip(breakindex+1).Take(inorder.Length).ToArray();
        root.left  = BuildTree(subleftpre,subleftin);
        root.right = BuildTree(subrightpre,subrightin);
        return root;
    }
}

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
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
        if(inorder.Length==0 || preorder.Length==0) return null;

            int ind = Array.IndexOf(inorder, preorder[0]);
        
            TreeNode root = new TreeNode(inorder[ind]);

            if(preorder.Length==1) return root;

            root.left = BuildTree( preorder.Skip(1).Take(ind+1).ToArray(), inorder.Skip(0).Take(ind).ToArray());
            
            root.right = BuildTree( preorder.Skip(ind+1).Take(preorder.Length).ToArray(), inorder.Skip(ind+1).Take(inorder.Length).ToArray());
                   
            return root;
            
    }

}

// caching positions of inorder[] indices using a Dictionary

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
    public TreeNode BuildTree(int[] preorder, int[] inorder) {
         Dictionary<int, int> inMap = new Dictionary<int, int>();
    
        for(int i = 0; i < inorder.Length; i++) {
            inMap[inorder[i]]= i;
        }

        TreeNode root = buildTree(preorder, 0, preorder.Length - 1, 0, inorder.Length - 1, inMap);
        return root;
    }

    public TreeNode buildTree(int[] preorder, int preStart, int preEnd,  int inStart, int inEnd, Dictionary<int, int> inMap) {
        if(preStart > preEnd || inStart > inEnd) return null;

        TreeNode root = new TreeNode(preorder[preStart]);
        int inRoot = inMap[root.val];
        int numsLeft = inRoot - inStart;

        root.left = buildTree(preorder, preStart + 1, preStart + numsLeft,  inStart, inRoot - 1, inMap);
        root.right = buildTree(preorder, preStart + numsLeft + 1, preEnd, inRoot + 1, inEnd, inMap);

        return root;
    }
}

// 103. Binary Tree Zigzag Level Order Traversal
// BFS 
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
    public IList<IList<int>> ZigzagLevelOrder(TreeNode root) {
        IList<IList<int>> ret = new List<IList<int>>();
        Queue<TreeNode> queue = new  Queue<TreeNode>();
        queue.Enqueue(root);
        int l = 0;
        while (queue.Count != 0) {
            int size = queue.Count;
            List<int> level = new List<int>();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.Peek();
                if (node != null) {
                    level.Add(node.val);
                    queue.Enqueue(node.left);
                    queue.Enqueue(node.right);
                }
            }
            if (level.Count!=0) {
                if (l % 2 == 1) {
                    level.Reverse();
                }
                ret.Add(level);
            }
            l++;
        }
        return ret;
    }
    
}
// DFS recursively
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
    public IList<IList<int>> ZigzagLevelOrder(TreeNode root) {
        List<IList<int>> ret = new List<IList<int>>();
     dfs(root, 0, ret);
     return ret;
 }
    private void dfs(TreeNode node, int l, List<IList<int>> ret) {
     if (node != null) {
         if (l == ret.Count) {
             IList<int> level = new List<int>();
             ret.Add(level);
         }
         if (l % 2 == 1) {
            ret[l].Insert(0, node.val);  // insert at the beginning
         } else {
            ret[l].Add(node.val);
         }
         dfs(node.left, l+1, ret);
         dfs(node.right, l+1, ret);
     }
 }
    
}

// Day 16 Tree

// 199. Binary Tree Right Side View
// DFS
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
    public IList<int> RightSideView(TreeNode root) {
        IList<int> result = new List<int>();   
        dfs(root, 0,result);
        return result;
    }
    public void dfs(TreeNode root, int depth, IList<int> result){
        
        if (root == null) {
            return ;
        }
        if (depth == result.Count){
            result.Add(root.val);
        }
            
        dfs(root.right, depth + 1,result);
        dfs(root.left, depth + 1,result);
    }
}

// BFS
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
    public IList<int> RightSideView(TreeNode root) {
        if(root is null)
        {
            return new List<int>();
        }
        Queue<TreeNode> bfs = new Queue<TreeNode>();
        IList<int> res = new List<int>();
        bfs.Enqueue(root);
        
        while(bfs.Count != 0)
        {
            int size = bfs.Count;
            for(int i = 0; i < size; i++)
            {
                TreeNode cur = bfs.Dequeue();
                 if(i == size - 1 )
                {
                    res.Add(cur.val);
                }
                if(cur.left != null)
                {
                    bfs.Enqueue(cur.left);
                }
                if(cur.right != null)
                {
                    bfs.Enqueue(cur.right);
                }
            }
           
        }
        return res;
    }
}

// 113. Path Sum II
// DFS
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
    public IList<IList<int>> PathSum(TreeNode root, int targetSum) {
      IList<IList<int>> ans = new List<IList<int>>();
        IList<int> curr = new List<int>();
        pathSum(root, targetSum, curr, ans);
        return ans;
    }
    public void pathSum(TreeNode root, int sum, IList<int> curr, IList<IList<int>> ans) {
            if(root==null) return;
            if(root.left==null && root.right==null) {
                if(root.val == sum) {
                    ans.Add( new List<int>(curr));
                    ans[ans.Count-1].Add(root.val);
                }
                return;
            }
            
            curr.Add(root.val);
            int new_sum = sum - root.val;
            pathSum(root.left, new_sum, curr, ans);
            pathSum(root.right, new_sum, curr, ans);
            curr.RemoveAt(curr.Count-1);
    }
}

// BFS
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
    public class Item {
  public TreeNode t { get; set; }
  public int i { get; set; }
  public List<int> l { get; set; }
}
    public IList<IList<int>> PathSum(TreeNode root, int targetSum) {
        if(root == null)
        {
            return new List<IList<int>>();
        }
        IList<IList<int>> res = new List<IList<int>>();
        List<Item> queue = new List<Item>();
       
        queue.Add(new Item {t = root, i = root.val, l = new List<int>() {root.val} });
        
        while(queue.Count != 0)
        {
            Item cur = queue[0]; queue.RemoveAt(0);
            if( cur.t.left == null && cur.t.right == null && cur.i == targetSum )
            {
                res.Add(cur.l);
            }
            if(cur.t.left != null)
            {
                queue.Add(new Item {t = cur.t.left, i = cur.i + cur.t.left.val, l = cur.l.Concat(new List<int>(){cur.t.left.val}).ToList() });              

            }
            if(cur.t.right != null)
            {   //Concat() returns a new sequence without modifying the original list.
                // AddRange() doesn't return any thing so won't work at here!
                queue.Add(new Item {t = cur.t.right, i = cur.i + cur.t.right.val, l = cur.l.Concat(new List<int>(){cur.t.right.val}).ToList()});
            }
           
        }
        return res;
    }
} 

// 450. Delete Node in a BST
/*
Steps:
1. Recursively find the node that has the same value as the key, 
while setting the left/right nodes equal to the returned subtree
2. Once the node is found, have to handle the below 4 cases
    node doesn't have left or right - return null
    node only has left subtree- return the left subtree
    node only has right subtree- return the right subtree
    node has both left and right - find the minimum value in the right subtree, 
    set that value to the currently found node,
    then recursively delete the minimum value in the right subtree
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
    public TreeNode DeleteNode(TreeNode root, int key) {
        if (root == null) return null;
        
        if (key > root.val) {
          root.right = DeleteNode(root.right, key);
            //return root;
        } 
        else if (key < root.val) {
          root.left = DeleteNode(root.left, key);
            //return root;
        } 
        else {
              // Find the min node in the right sub tree
            if (root.left != null && root.right != null) 
            {
                TreeNode min = root.right;
                while (min.left != null) 
                {
                    min = min.left;
                }

                root.val = min.val;
                root.right = DeleteNode(root.right, min.val);
                //return root;
            } 
            else {

                if(root.left==null){
                    return root.right;
                }
                else if(root.right==null){
                    return root.left;
                }
                
               
            }
        }    
        return root;
    }
}

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
    public TreeNode DeleteNode(TreeNode root, int key) {
        
        if (root != null) { //We frecursively call the function until we find the target node
                if (key > root.val) { 
              root.right = DeleteNode(root.right, key);
            } 
            else if (key < root.val) {
              root.left = DeleteNode(root.left, key);
            } 
            else {
                if(root.left == null && root.right == null) return null;          //No child condition
                if (root.left==null || root.right==null)            //One child contion -> replace the node with it's child
                    return root.left!=null ? root.left : root.right; 
                                                                //Two child condition  
                TreeNode min = root.right;                      //(or) TreeNode temp = root.left;
                    while (min.left != null)                    //while(temp.right != NULL) temp = temp.right;
                    {                                           //root.val = temp.val; 
                        min = min.left;                         //root.left = deleteNode(root.left, temp.val);	
                    }                                           //

                    root.val = min.val;
                    root.right = DeleteNode(root.right, min.val);

            }
        }    
        
        
        return root;
    }
}

// Day 17 Tree

// 230. Kth Smallest Element in a BST
/*
 Solution : Inorder Traversal
We traverse inorder in BST (Left - Root - Right) then we have sortedArr as non-decreasing sorted array af elements in BST.
The result is the kth element in our sortedArr.

Complexity
Time: O(N)
Space: O(N)
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
    public int KthSmallest(TreeNode root, int k) {
        List<int> nums = inorder(root, new List<int>());
        return nums[k - 1];
  }
public List<int> inorder(TreeNode root, List<int> arr) {
   if (root == null) return arr;
    inorder(root.left, arr);
    arr.Add(root.val);
    inorder(root.right, arr);
    return arr;
  }
 
}

/*
 Solution : Using Stack as Iterator
We use idea from 173. Binary Search Tree Iterator to iterate elements in order in O(H) in Space Comlpexity.
The idea is that we iterate through the BST, we pop from smallest elements and we just need to pop k times to get the k_th smallest element.

Complexity
Time: O(H + k), where H is the height of the BST.
Space: O(H)
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
    public int KthSmallest(TreeNode root, int k) {
        Stack<TreeNode> stack = new Stack<TreeNode>();
        pushLeft(stack, root);
        while (--k > 0) {
            pushLeft(stack, stack.Pop().right);
        }
        return stack.Pop().val;
  }
public void pushLeft(Stack<TreeNode> stack, TreeNode root) {
        while (root != null) {
            stack.Push(root);
            root = root.left;
        }
    }
 
}

// 173. Binary Search Tree Iterator
/*
with what we're supposed to support here:

1.    BSTIterator i = new BSTIterator(root);
2.    while (i.hasNext())
3.        doSomethingWith(i.next());
You can see they already have the exact same structure:

Some initialization.
A while-loop with a condition that tells whether there is more.
The loop body gets the next value and does something with it.
So simply put the three parts of that iterative solution into our three iterator methods:
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
public class BSTIterator {

    private TreeNode visit;
    private Stack<TreeNode> stack;
    
    public BSTIterator(TreeNode root) {
        visit = root;
        stack = new Stack<TreeNode>();
    }
    
    public int Next() {
        while (visit != null) {
            stack.Push(visit);
            visit = visit.left;
        }
        TreeNode next = stack.Pop();
        visit = next.right;
        return next.val;
    }
    
    public bool HasNext() {
        return visit != null || stack.Count != 0;
        
    }
    
}

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.Next();
 * bool param_2 = obj.HasNext();
 */

// Day 18 Tree

// 236. Lowest Common Ancestor of a Binary Tree
// Recursive Approach
// Complexity Analysis

// Time Complexity: O(N), where NN is the number of nodes in the binary tree.
// In the worst case we might be visiting all the nodes of the binary tree.

// Space Complexity: O(N). This is because the maximum amount of space utilized by the recursion stack would be N
// since the height of a skewed binary tree could be N.
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int x) { val = x; }
 * }
 */
public class Solution {
    public TreeNode LowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode l = LowestCommonAncestor(root.left, p, q);
        TreeNode r = LowestCommonAncestor(root.right, p, q);
        if (l == null || r == null) return l == null ? r : l;
        return root;
    }
}

// 297. Serialize and Deserialize Binary Tree

// DFS - Serialize and Deserialize in Pre Order Traversal
// Complexity
// Time: O(N), where N <= 10^4 is number of nodes in the Binary Tree.
// Space: O(N)

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left;
 *     public TreeNode right;
 *     public TreeNode(int x) { val = x; }
 * }
 */
public class Codec {

    // Encodes a tree to a single string.
    public string serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        preOrderTraverse(root, sb);
        sb.Length--; // delete the last redundant comma ","
        return sb.ToString();
    }

   public void preOrderTraverse(TreeNode root, StringBuilder sb) {
        if (root == null) {
            sb.Append("null,");
            return;
        }
        sb.Append(root.val);
        sb.Append(",");
        preOrderTraverse(root.left, sb);
        preOrderTraverse(root.right, sb);
    }
    // Decodes your encoded data to tree.
    public TreeNode deserialize(string data) {
        nodes = data.Split(",");
        return dfs();
    }
    
    int i = 0;
    String[] nodes;
    public TreeNode dfs() {
        if (i == nodes.Length) return null;
        String val = nodes[i];
        i += 1;
        if (val.Equals("null")) return null;
        TreeNode root = new TreeNode(Int32.Parse(val));
        root.left = dfs();
        root.right = dfs();
        return root;
    }
    
}

// Your Codec object will be instantiated and called as such:
// Codec ser = new Codec();
// Codec deser = new Codec();
// TreeNode ans = deser.deserialize(ser.serialize(root));

// Day 19 Graph

// 997. Find the Town Judge
/*
Solution : Count inDegree, outDegree of vertices

We count inDegree, outDegree of vertices.
Let degree[i] is the result of inDegree[i] + outDegree[i].
If degree[i] == N - 1 then i is the town judge.

node with degree (in_degree – out_degree) N – 1 is the judge.

Complexity
Time: O(M + N), where M <= 10^4 is length of trust array, N <= 1000 is number of people.
Space: O(N)

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
// Nodes with no In-Degree
/*
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

// 841. Keys and Rooms

// iterative DFS
// Keys and Rooms [stack implementation]
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

// Solution : Recursive DFS
// Check whether the entire graph is a connected component.
// Complexity:
// Time: O(M + N), where M <= 3000 is total number of edges which is sum(rooms[i].length), 
// N <= 1000 is number of vertices.
// Space: O(N)
public class Solution {
public bool CanVisitAllRooms(IList<IList<int>> rooms) {
        List<int> visited = new List<int>();
        dfs(rooms, 0, visited);
        return visited.Count == rooms.Count;
  }
public void dfs(IList<IList<int>> rooms, int cur, List<int> visited) {
        if (visited.Contains(cur)) return;
        visited.Add(cur);
        foreach (int nxt in rooms[cur])
            dfs(rooms, nxt, visited);
  }
}

// Day 20 Heap (Priority Queue)

// 215. Kth Largest Element in an Array

// an iterative version of QuickSelect 
// only shrink the range between l and r but never change k
public class Solution {
    public int FindKthLargest(int[] nums, int k) {
        k = nums.Length - k; // convert to index of k largest
        int l = 0, r = nums.Length - 1;
        while (l <= r) {
            int i = l; // partition [l,r] by A[l]: [l,i]<A[l], [i+1,j)>=A[l]
            // use quick sort's idea
            // put nums that are <= pivot to the left
            // put nums that are  > pivot to the right
            for (int j = l + 1; j <= r; j++)
                if (nums[j] < nums[l]) swap(nums, j, ++i);
            swap(nums, l, i);

            // prefix (++i) the variable is incremented and then used whereas postfix (i++) the variable is used and then incrmented.
            
            // count the nums that are > pivot from high
            // pivot is too small, so it must be on the right
            if (k < i) r = i - 1;
            // pivot is too big, so it must be on the left
            else if (k > i) l = i + 1;
            // pivot is the one!
            else return nums[i];
        }
        return -1; // k is invalid
    }
    
    private void swap(int[] nums, int a, int b) {
        int temp = nums[a];
        nums[a] = nums[b];
        nums[b] = temp;
    }
}

// 347. Top K Frequent Elements
// use an array to save numbers into different bucket whose index is the frequency
public class Solution {
    public int[] TopKFrequent(int[] nums, int k) {
        Dictionary<int, int> map = new Dictionary<int, int>();
        foreach(int n in nums){
            map[n]= map.GetValueOrDefault(n,0) + 1;
        }
        
        // corner case: if there is only one number in nums, we need the bucket has index 1.
        List<int>[] bucket = new List<int>[nums.Length+1];
        foreach(int n in map.Keys){
            int freq = map[n];
            if(bucket[freq]==null)
                bucket[freq] = new List<int>();
            bucket[freq].Add(n);
        }
        
        List<int> res = new List<int>();
        for(int i=bucket.Length-1; i>0 && k>0; i--){
            if(bucket[i]!=null){
                List<int> list = bucket[i]; 
                res.AddRange(list.ToArray());
                k-= list.Count;
            }
        }
       
        return res.ToArray();
    }
}

// use maxHeap. Put entry into maxHeap so we can always poll a number with largest frequency
public class Solution {
    public int[] TopKFrequent(int[] nums, int k) {
       Dictionary<int, int> map = new Dictionary<int, int>();
        foreach(int n in nums){
            map[n]= map.GetValueOrDefault(n,0) + 1;
        }
           
        PriorityQueue<int, int> maxHeap = 
                         new PriorityQueue<int,int>(Comparer<int>.Create((a, b) => b - a));
        foreach(var item in map){
            maxHeap.Enqueue(item.Key,item.Value);
        }
        
        List<int> res = new List<int>();
        while(res.Count<k){
            maxHeap.TryDequeue(out int item, out int priority);
            res.Add(item);
        }
        return res.ToArray();
    
    }
}

// Day 21 Heap (Priority Queue)
// 451. Sort Characters By Frequency
// Dictionary + Sort: 
// We use Dictionary to count the occurence and sort the entries based on the occurence,
// then build the string.
// Time Complexity: O(nlogn), since we sort the characters
// Space Complexity: O(n)
public class Solution {
    public string FrequencySort(string s) {
        int n = s.Length;
        // Count the occurence on each character
        Dictionary<char, int> cnt = new Dictionary<char, int>();
        foreach (char c in s.ToCharArray())  
            cnt[c]= cnt.GetValueOrDefault(c,0) + 1;

        // Sorting
        List<char> chars = cnt.Keys.ToList().OrderByDescending(i => cnt[i]).ToList();
                
        // Build string
        StringBuilder ans = new StringBuilder();
        foreach (char c in chars) {
            for (int i = 0; i < cnt[c]; i++) {
                ans.Append(c);
            }
	    }
        return ans.ToString();
    }
}
// Bucket Sort Solution
// O(n) 
// The logic is very similar to NO.347 and here we just use a map a count 
// and according to the frequency to put it into the right bucket.
// Then we go through the bucket to get the most frequently character
// and append that to the final stringbuilder.
public class Solution {
    public string FrequencySort(string s) {
        int n = s.Length;
        Dictionary<char, int> cnt = new Dictionary<char, int>();
        foreach (char c in s.ToCharArray())  
            cnt[c]= cnt.GetValueOrDefault(c,0) + 1;

        List<char>[]  bucket = new List<char>[n+1] ;
        foreach (var (c, f) in cnt){
            if (bucket[f] == null) 
                bucket[f] = new List<char>() ;
            bucket[f].Add(c);
        }
            
        
        StringBuilder ans = new StringBuilder();
        for (int freq = n; freq >= 1; --freq)
            if (bucket[freq] != null)
                foreach (char c in bucket[freq]) 
                    for (int i = 0; i < freq; i++)
                        ans.Append(c);
        return ans.ToString();
    }
}
// using PriorityQueue: 
// O(n) ignore logm since m is the distinguish character, can be O(1) since only 26 letters.
// So the overall time complexity should be O(n), the same as the buck sort with less memory use.
// There is a follow up if you are interested,
// when same frequency we need to maintain the same sequence as the character show in the original string,
// the solution is add a index as a secondary sort if the frequency is same, 

// Dictionary + Heap(Maxheap): 
// We use HashTable to count the occurence and build the heap based on the occurence, then build the string.
public class Solution {
    public string FrequencySort(string s) {
        // Count the occurence on each character
        Dictionary<char, int[]> map = new Dictionary<char, int[]>();
        for (int i = 0; i <s.Length; i++) {
            char c = s[i];
            if (!map.ContainsKey(c)) map[c] = new int[2]{1,i};
            else {
                map[c][0]++;
            }
        }
        
        // Build heap
        PriorityQueue<char, int[]> pq = new PriorityQueue<char, int[]>(Comparer<int[]>.Create((x, y) => x[0] == y[0] ?  x[1] - y[1] : y[0] - x[0]));     
        foreach(var item in map)
        {
            pq.Enqueue(item.Key,item.Value);
        }
        
        // Build string
        StringBuilder sb = new StringBuilder();    
        while (pq.TryDequeue(out char item, out int[] priority))
        {
            for (int i = 0; i < priority[0]; i++)
                sb.Append(item);
        }
        
        return sb.ToString();
	
    }
}
// Solution : Counter & Sorting String S
// Complexity
// Time: O(NlogN), where N <= 5 * 10^5 is the length of string s.
// Space:
// C++: O(logN) it's the stack memory of introsort in C++.
// Python: O(N)
public class Solution {
    public string FrequencySort(string s) {
        int[] cnt = new int[128];
        foreach (char c in s) cnt[c] += 1;

        return String.Concat(s.OrderByDescending(c => cnt[c]).ThenBy(c => c));
        //there's also ThenByDescending() 
    }
}
// 973. K Closest Points to Origin

// The very naive and simple solution is sorting the all points by their distance to the origin point directly,
// then get the top k closest points. We can use the sort function and the code is very short.

// Theoretically, the time complexity is O(NlogN).

// The advantages of this solution are short, intuitive and easy to implement.
// The disadvatages of this solution are not very efficient and have to know all of the points previously,
// and it is unable to deal with real-time(online) case, it is an off-line solution.
public class Solution {
    public int[][] KClosest(int[][] points, int k) {
        Array.Sort(points, (p1, p2)  => (p1[0] * p1[0] + p1[1] * p1[1]).CompareTo(p2[0] * p2[0] + p2[1] * p2[1]));
        
        return points[0..k];

        // Copy array from source to destination
        /*
        int[][] outputArray = new int[k][];
        Array.Copy(points, 0, outputArray, 0, k);
        return outputArray;*/
    }
}
// Sort all points and return K first, O(NlogN)
public class Solution {
    public int[][] KClosest(int[][] points, int k) {
        //Array.Sort(points, (p1, p2)  => (p1[0] * p1[0] + p1[1] * p1[1]).CompareTo(p2[0] * p2[0] + p2[1] * p2[1])); // Quick
        points = points.OrderBy(p => p[0] * p[0] + p[1] * p[1]).ToArray(); // Slow 
        
        return points[0..k];

        // Copy array from source to destination
        /*
        int[][] outputArray = new int[k][];
        Array.Copy(points, 0, outputArray, 0, k);
        return outputArray;*/
    }
}
/*
The second solution is based on the first one. We don't have to sort all points.
Instead, we can maintain a max-heap with size K.
Then for each point, we add it to the heap.
Once the size of the heap is greater than K, we are supposed to extract one from the max heap to ensure the size of the heap is always K.
Thus, the max heap is always maintain top K smallest elements from the first one to crruent one. Once the size of the heap is over its maximum capacity, it will exclude the maximum element in it, since it can not be the proper candidate anymore.

Theoretically, the time complexity is O(NlogK), but pratically, the real time it takes on leetcode is 134ms.

The advantage of this solution is it can deal with real-time(online) stream data. It does not have to know the size of the data previously.
The disadvatage of this solution is it is not the most efficient solution.
*/

public class Solution {
    public int[][] KClosest(int[][] points, int k) {
        PriorityQueue<int[],int[]> maxHeap = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => (b[0] * b[0] + b[1] * b[1]) - (a[0] * a[0] + a[1] * a[1])));
        foreach (int[] p in points) {
            maxHeap.Enqueue(p,p);
            if (maxHeap.Count > k)
                maxHeap.Dequeue();
        }
        int[][] ans = new int[maxHeap.Count][];
        while (k > 0)
            ans[--k] = maxHeap.Dequeue();
        return ans;
    }
}

public class Solution {
    public int[][] KClosest(int[][] points, int k) {
        PriorityQueue<int[],int[]> maxHeap = new PriorityQueue<int[],int[]>(Comparer<int[]>.Create((a, b) => (b[0] * b[0] + b[1] * b[1]) - (a[0] * a[0] + a[1] * a[1])));
        foreach (int[] p in points) {
            maxHeap.Enqueue(p,p);
            if (maxHeap.Count > k)
                maxHeap.Dequeue();
        }
        int[][] ans = new int[maxHeap.Count][];
        int i = 0;
        while (maxHeap.Count > 0)
            ans[i++] = maxHeap.Dequeue();//slightly different here
        return ans;

    }
}
/*
The last solution is based on quick sort, we can also call it quick select. 
In the quick sort, we will always choose a pivot to compare with other elements. 
After one iteration, we will get an array that all elements smaller than the pivot are on the left side of the pivot 
and all elements greater than the pivot are on the right side of the pviot (assuming we sort the array in ascending order). 
So, inspired from this, each iteration, we choose a pivot and then find the position p the pivot should be. 
Then we compare p with the K, if the p is smaller than the K, meaning the all element on the left of the pivot are all proper candidates
 but it is not adequate, we have to do the same thing on right side, and vice versa. 
If the p is exactly equal to the K, meaning that we've found the K-th position. 
Therefore, we just return the first K elements, since they are not greater than the pivot.

Theoretically, the average time complexity is O(N) , but just like quick sort, 
 in the worst case, this solution would be degenerated to O(N^2),

The advantage of this solution is it is very efficient.
The disadvatage of this solution are it is neither an online solution nor a stable one.
 And the K elements closest are not sorted in ascending order.
*/
public class Solution {
    public int[][] KClosest(int[][] points, int k) {
        int len =  points.Length, l = 0, r = len - 1;
    while (l <= r) {
        int mid = helper(points, l, r);
        if (mid == k) break;
        if (mid < k) {
            l = mid + 1;
        } else {
            r = mid - 1;
        }
    }
        return points[0..k];// 1.
        //source.Skip(100).Take(100).ToArray();
        // return points.Take(k).ToArray(); // 2. LinQ works but slow
        
        // 3. Manually Copy array from source to destination
        // int[][] outputArray = new int[k][];
        // Array.Copy(points, 0, outputArray, 0, k);
        // return outputArray;
}

    private int helper(int[][] A, int l, int r) {
        int[] pivot = A[l];
        while (l < r) {
            while (l < r && compare(A[r], pivot) >= 0) r--;
            A[l] = A[r];
            while (l < r && compare(A[l], pivot) <= 0) l++;
            A[r] = A[l];
        }
        A[l] = pivot;
        return l;
    }

    private int compare(int[] p1, int[] p2) {
        return p1[0] * p1[0] + p1[1] * p1[1] - p2[0] * p2[0] - p2[1] * p2[1];
    }
}