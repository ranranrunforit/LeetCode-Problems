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