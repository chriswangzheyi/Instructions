package com.wzy.article.service;



import com.wzy.article.pojo.Article;
import java.util.List;

/**
 * 文章service接口
 */
public interface ArticleService {

    public List<Article> findAll();

    public Article findById(Integer id);

    public void add(Article article);

    public void update(Article article);

    public void deleteById(Integer id);
}
