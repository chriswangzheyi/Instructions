# 491 · Palindrome Number

## 题目

Check a positive number is a palindrome or not.

A palindrome number is that if you reverse the whole number you will get exactly the same number.

Example 1:

	Input:11
	Output:true
Example 2:

	Input:1232
	Output:false
	Explanation:
	1232!=2321
	
## 代码

	class Solution:
	    """
	    @param num: a positive number
	    @return: true if it's a palindrome or false
	    """
	    def is_palindrome(self, num: int) -> bool:
	        # write your code here
	        return str(num) == str(num)[::-1]