#40 · Implement Queue by Two Stacks

## 题目

As the title described, you should only use two stacks to implement a queue's actions.

The queue should support push(element), pop() and top() where pop is pop the first(a.k.a front) element in the queue.

Both pop and top methods should return the value of first element.

Example 1:

Input:

	Queue Operations = 
	    push(1)
	    pop()    
	    push(2)
	    push(3)
	    top()    
	    pop()  
Output:

	1
	2
	2
Explanation:

Both pop and top methods should return the value of the first element.

Example 2:

Input:

	Queue Operations = 
	    push(1)
	    push(2)
	    push(2)
	    push(3)
	    push(4)
	    push(5)
	    push(6)
	    push(7)
	    push(1)
Output:

	[]
	
## 代码

	class MyQueue:
	    
	    def __init__(self):
	        self.stack1 = []
	        self.stack2 = []
	    """
	    @param: element: An integer
	    @return: nothing
	    """
	    def push(self, element):
	        self.stack1.append(element)
	    """
	    @return: An integer
	    """
	    def pop(self):
	        if len(self.stack2) == 0:
	            self.move()
	        return self.stack2.pop()
	    """
	    @return: An integer
	    """
	    def top(self):
	        if len(self.stack2) == 0:
	            self.move()
	        return self.stack2[-1]
	    
	    # 从1号栈转移到2号栈
	    def move(self):
	        while len(self.stack1) > 0:
	            self.stack2.append(self.stack1.pop())
	            
## 解释

###解题思路
先考虑只有一个栈的时候，由于栈的先入后出特性FILO，栈中的元素的顺序是反的，我们无法直接访问栈底的元素。但是当把1号栈中所有元素依次弹出并压入到2号栈中，2号栈顶的元素就变成了原来1 号栈的栈底，即正序。所以我们要提取元素时，只需从2号栈提取即可。

但是由于2号栈中栈顶元素是最先加入队列的元素，所以只有当2号栈为空时，才能将1号栈中所有元素加入到2号栈中。

举例说明：

首先我们有一个主要栈stack1：[1,2,3) ，以下所有栈的表示方式中，圆括号 ')' 均为栈顶。 那么stack1的出栈顺序为3-2-1，其中 1 为我们要找到的元素，也就是队首。

我们需要借助一个辅助栈stack2：[)，将stack1中的元素依次放到stack2中：stack2 [3,2,1)。这时我们发现stack2的栈顶就是我们要找的元素，弹出即可。

此时我们再向主要栈stack1中压入 4 和 5。两个栈状态：stack1 [4,5) 、stack2 [3,2)。

现在我们需要队首的话，应该先弹出辅助栈stack2的栈顶。

如果此时辅助栈空，我们就要执行之前转移的操作，将stack1的所有元素压入stack2，然后弹出stack2的栈顶即可。

###代码思路
定义move()，操作是将元素从1号栈转移到2号栈。当要提取元素，且2号栈为空时，调用move()。

###复杂度分析

####时间复杂度
每个元素最多会别push，pop，move一次，每个操作的均摊时间复杂度为O(1)。

####空间复杂度
假设一共操作了N次push，空间复杂度为O(N)。