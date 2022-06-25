# 1300 · Bash Game

You are playing the following game with your friend: There is a pile of stones on the table, each time one of you take turns to remove 1 to 3 stones. The one who removes the last stone will be the winner. You will take the first turn to remove stones.

Both of you are very clever and have optimal strategies for the game. Write a function to determine whether you can win the game given the number of stones.

For example, if there are 4 stones, then you will never win the game: no matter 1, 2, or 3 stones you remove, the last stone will always be removed by your friend.

Example 1：

	Input：n = 4 
	Output：False
	Explanation：Take 1, 2 or 3 first, the other party will take the last one
Example 2：

	Input：n = 5 
	Output：True
	Explanation：Take 1 first，Than，we can win the game
	
##  代码

	class Solution:
	    """
	    @param n: an integer
	    @return: whether you can win the game given the number of stones in the heap
	    """
	    def can_win_bash(self, n: int) -> bool:
	        # Write your code here
	         return n % 4 != 0
	         
## 解释

方法：数学推理
思路与算法

让我们考虑一些小例子。显而易见的是，如果石头堆中只有一块、两块、或是三块石头，那么在你的回合，你就可以把全部石子拿走，从而在游戏中取胜；如果堆中恰好有四块石头，你就会失败。因为在这种情况下不管你取走多少石头，总会为你的对手留下几块，他可以将剩余的石头全部取完，从而他可以在游戏中打败你。因此，要想获胜，在你的回合中，必须避免石头堆中的石子数为 44 的情况。

我们继续推理，假设当前堆里只剩下五块、六块、或是七块石头，你可以控制自己拿取的石头数，总是恰好给你的对手留下四块石头，使他输掉这场比赛。但是如果石头堆里有八块石头，你就不可避免地会输掉，因为不管你从一堆石头中挑出一块、两块还是三块，你的对手都可以选择三块、两块或一块，以确保在再一次轮到你的时候，你会面对四块石头。显然我们继续推理，可以看到它会以相同的模式不断重复 n = 4, 8, 12, 16, \ldotsn=4,8,12,16,…，基本可以看出如果堆里的石头数目为 44 的倍数时，你一定会输掉游戏。

如果总的石头数目为 44 的倍数时，因为无论你取多少石头，对方总有对应的取法，让剩余的石头的数目继续为 44 的倍数。对于你或者你的对手取石头时，显然最优的选择是当前己方取完石头后，让剩余的石头的数目为 44 的倍数。假设当前的石头数目为 xx，如果 xx 为 44 的倍数时，则此时你必然会输掉游戏；如果 xx 不为 44 的倍数时，则此时你只需要取走 x \bmod 4xmod4 个石头时，则剩余的石头数目必然为 44 的倍数，从而对手会输掉游戏。