# 415 · Valid Palindrome

## 题目

Given a string, determine if it is a palindrome, considering only letters are considered and case is ignored

样例
Example 1:

Input: "A man, a plan, a canal: Panama"

Output: true
	
Explanation: "amanaplanacanalpanama"
	
Example 2:

Input: "race a car"

Output: false

Explanation: "raceacar"


## 代码

	class Solution:
	    """
	    @param s: A string
	    @return: Whether the string is a valid palindrome
	    """
	    def is_palindrome(self, s: str) -> bool:
	        # write your code here
	        left, right = 0, len(s) - 1
	
	        while left < right:
	            # 判断是否为数字，不为数字则指针移动
	            while left < right and not self.is_valid(s[left]):
	                left +=1
	
	            # 判断是否为数字，不为数字则指针移动
	            while left < right and not self.is_valid(s[right]):
	                right -=1
	
	            # 判断两个数字是否相同
	            if left < right and s[left].lower() != s[right].lower():
	                return False
	
	            # 同时往后移动指针
	            left += 1
	            right -= 1
	        return True
	
	    def is_valid(self,char):
	        return char.isdigit() or char.isalpha()