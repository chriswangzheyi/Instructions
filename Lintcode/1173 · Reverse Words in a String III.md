# 1173 · Reverse Words in a String III

## 题目

Given a string, you need to reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.

样例

	Input: "Let's take LeetCode contest"
	Output: "s'teL ekat edoCteeL tsetnoc"

## 代码

	class Solution:
	    """
	    @param s: a string
	    @return: reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order
	    """
	    def reverse_words(self, s: str) -> str:
	        # Write your code here
	        ans = []
	        for word in s.split(" "):
	          ans.append(word[::-1])
	        
	        return " ".join(ans)