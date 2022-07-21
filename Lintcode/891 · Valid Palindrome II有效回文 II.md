# 891 · Valid Palindrome II

## 题目

Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome

Example 1:

	Input: s = "aba"
	Output: true
	Explanation: Originally a palindrome.
	
Example 2:

	Input: s = "abca"
	Output: true
	Explanation: Delete 'b' or 'c'.
	
Example 3:

	Input: s = "abc"
	Output: false
	Explanation: Deleting any letter can not make it a palindrome.
	
## 代码 （双指针）

	class Solution:
	    """
	    @param s: a string
	    @return: whether you can make s a palindrome by deleting at most one character
	    """
	    def valid_palindrome(self, s: str) -> bool:
	        # Write your code here
	        if s is None:
	            return False
	
	        left, right = self.findDifference(s, 0, len(s)-1)
	
	        if left >= right:
	            return True
	
	        return self.isPalindrome(s, left + 1, right) or \
	            self.isPalindrome(s, left, right - 1)
	
	    def isPalindrome(self, s, left, right):
	        left,right = self.findDifference(s, left, right)
	        return left >= right
	
	    def findDifference(self, s, left, right):
	        while left < right:
	            if s[left] != s[right]:
	                return left, right
	            left += 1
	            right -= 1
	        return left, right