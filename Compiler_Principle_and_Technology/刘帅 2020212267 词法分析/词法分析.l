%{ 
    #include <stdio.h>  
    #include <stdlib.h>   
    int count = 0;  
%}   
  
/*第一小部分以符号%{和%}包裹，里面为以C语法写的一些定义和声明：例如，文件包含，宏定义，常数定义，
全局变量及外部变量定义，函数声明等。这一部分被Lex翻译器处理后会全部拷贝到文件lex.yy.c中 */
  
delim [" "\n\t\r]   
whitespace {delim}+   
operator \+|-|\*|\/|:=|>=|<=|#|=|<<|>>|\+\+|\<|\>|\{|\}
reservedWord int|include|main|return|using|if|namespace|cout|cin|std|iostream
delimiter [,\.;\(\)\"]
constant ([0-9])+
identfier [A-Za-z]([A-Za-z][0-9])*  
/* 词法规则 正规定义和状态定义 */

%%   
{reservedWord} {count++;printf("%d\t(keyword,%s)\n",count,yytext);}  
\"[^\"]*\" {count++;printf("%d\t(count,%s)\n",count,yytext);}
{operator} { count++;printf("%d\t(operator,%s)\n",count,yytext); }  
{delimiter} {count++;printf("%d\t(delimiter,%s)\n",count,yytext);}  
{constant} {count++;printf("%d\t(constant,%s)\n",count,yytext);}  
{identfier} {count++;printf("%d\t(id,%s)\n",count,yytext);}   
{whitespace} { /* do    nothing*/ } 
%%

int main()
{
	yyin = fopen("input.txt","r");
		yylex();
	fclose(yyin);
}
int yywrap()
{
	return 1;
}
/*若yywrap()返回0，则继续扫描
若返回1，则返回报告文件结尾的0标记。
由于词法分析器总会调用yywrap，因此辅助函数中最好提供yywrap，
如果不提供，则在用C编译器编译lex.yy.c时，需要链接相应的库，库中会给出标准的yywrap函数（标准函数返回1） */