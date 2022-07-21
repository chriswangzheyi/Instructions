# 423 · Valid Parentheses

## 题目

Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.

Example 1:

	Input: "([)]"
	Output: False
Example 2:

	Input: "()[]{}"
	Output: True
	
## 代码

	class Solution:
	    """
	    @param s: A string
	    @return: whether the string is a valid parentheses
	    """
	    def is_valid_parentheses(self, s: str) -> bool:
	        # write your code here
	        stack = []
	        for ch in s:
	            # 压栈
	            if ch == '{' or ch == '[' or ch == '(':
	                stack.append(ch)
	            else:
	                # 栈需非空
	                if not stack:
	                    return False
	                # 判断栈顶是否匹配
	                if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
	                    return False
	                # 弹栈
	                stack.pop()
	        return not stack
	        

## 解释

作为括号左边的元素会放在stack中，右边只要有反括号则对stack做pop操作。如果最后stack为空，则表示有效，如果stack不为空（not stack）则表示无效。


