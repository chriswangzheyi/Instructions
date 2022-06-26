# 53 · Reverse Words in a String

## 题目

Given an input string, reverse the string word by word.

Example 1:

Input:

	s = "the sky is blue"
Output:

	"blue is sky the"
Explanation:

return a reverse the string word by word.
Example 2:

Input:

	s = "hello world"
Output:

	"world hello"
Explanation:

return a reverse the string word by word.

## 代码

	class Solution:
	    """
	    @param s: A string
	    @return: A string
	    """
	    def reverse_words(self, s: str) -> str:
	        # write your code here
	         return " ".join(reversed(s.split()))
	         
## 解释

* 使用 split 将字符串按空格分割成字符串数组；
* 使用 reverse 将字符串数组进行反转；
* 使用 join 方法将字符串数组拼成一个字符串。