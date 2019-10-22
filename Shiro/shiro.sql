/*==============================================================*/
/* DBMS name:      MySQL 5.0                                    */
/* Created on:     2019/10/21 14:56:25                          */
/*==============================================================*/


/*==============================================================*/
/* Table: t_permission                                          */
/*==============================================================*/
create table t_permission
(
   id                   int(10) not null,
   name                 varchar(50),
   permission           varchar(50),
   primary key (id)
);

/*==============================================================*/
/* Table: t_role                                                */
/*==============================================================*/
create table t_role
(
   id                   int(10) not null,
   name                 varchar(50),
   permission           varchar(50),
   primary key (id)
);

/*==============================================================*/
/* Table: t_role_permission                                     */
/*==============================================================*/
create table t_role_permission
(
   role_id              int(10),
   permission_id        int(10)
);

/*==============================================================*/
/* Table: t_user                                                */
/*==============================================================*/
create table t_user
(
   id                   int(10) not null,
   name                 varchar(50),
   password             varchar(50),
   primary key (id)
);

/*==============================================================*/
/* Table: t_user_role                                           */
/*==============================================================*/
create table t_user_role
(
   user_id              int(10),
   role_id              int(10)
);

alter table t_role_permission add constraint FK_Reference_3 foreign key (role_id)
      references t_role (id) on delete restrict on update restrict;

alter table t_role_permission add constraint FK_Reference_4 foreign key (permission_id)
      references t_permission (id) on delete restrict on update restrict;

alter table t_user_role add constraint FK_Reference_1 foreign key (user_id)
      references t_user (id) on delete restrict on update restrict;

alter table t_user_role add constraint FK_Reference_2 foreign key (role_id)
      references t_role (id) on delete restrict on update restrict;

