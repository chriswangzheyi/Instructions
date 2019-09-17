package com.wzy.article.controller;


import com.wzy.article.pojo.Article;
import com.wzy.article.pojo.Result;
import com.wzy.article.service.ArticleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;


/**
 * 文章Controller
 */
@RestController
@RequestMapping("/article")
public class ArticleController {

    @Autowired
    private ArticleService articleService;


    @Autowired
    private HttpServletRequest request;
    /**
     * 查询所有
     */
    @RequestMapping(method = RequestMethod.GET)
    public Result findAll(){
        return new Result(true,"查询成功:"+request.getLocalAddr(),articleService.findAll());
    }

    /**
     * 查询一个
     */
    @RequestMapping(value = "/{id}",method = RequestMethod.GET)
    public Result findById(@PathVariable Integer id){
        return new Result(true,"查询成功",articleService.findById(id));
    }

    /**
     * 添加
     */
    @RequestMapping(method = RequestMethod.POST)
    public Result add(@RequestBody Article article){
        articleService.add(article);
        return new Result(true,"添加成功");
    }

    /**
     * 修改
     */
    @RequestMapping(value = "/{id}",method = RequestMethod.PUT)
    public Result update(@RequestBody Article article,@PathVariable Integer id){
        article.setId(id);
        articleService.update(article);
        return new Result(true,"修改成功");
    }

    /**
     * 删除
     */
    @RequestMapping(value = "/{id}",method = RequestMethod.DELETE)
    public Result deleteById(@PathVariable Integer id){
        articleService.deleteById(id);
        return new Result(true,"删除成功");
    }
}
