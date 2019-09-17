package com.wzy.article.dao;


import com.wzy.article.pojo.Article;
import org.springframework.data.jpa.repository.JpaRepository;

/**
 * 文章dao
 */
public interface ArticleDao extends JpaRepository<Article,Integer>{

}
