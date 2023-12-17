%{
  #include <stdio.h>
  extern int yylex(void);
   void yyerror(char *);
   #define YYSTYPE double
%}


%token NUMBER /*由yylex通过lex1.l传过来的词法规则*/
%left '+' '-'
%left '*' '/'
 
%%
L : E '$'  {printf("Result=%f\n",$1);system("pause");return 0;}
;
E : T        {printf("Reduce by E-->T\n");$$=$1;}
    | E'+' T    {printf("Reduce by E-->E+T\n");$$=$1+$3;} /* 规则右部的文法相加赋值给左部非终结符属性 */
    | E'-' T    {printf("Reduce by E-->E-T\n");$$=$1-$3;}
;

T : F         {printf("Reduce by T-->F\n");$$=$1;}
    | T'*' F    {printf("Reduce by T-->T*F\n");$$=$1*$3;}
    | T'/' F    {printf("Reduce by T-->T/F\n");$$=$1/$3;}
;

F : '('  E  ')' { printf("Reduce by F-->(E)\n");$$ = $2; }
    | NUMBER {printf("Reduce by F-->num\n"); $$ = $1; }
;
%%
 

int main()
{
    printf("Please input an expression\n");
    yyparse();
	/*yyparse 函数调用一个扫描函数（即词法分析程序）yylex。
	yyparse 每次调用 yylex() 就得到一个二元式的记号<token,attribute> 。
	由 yylex() 返回的记号(如下 NUMBER 等)，必须事先在 YACC 源程序的说明部分用%token说明，
	该记号的属性值必须通过 YACC 定义的变量 yylval 传给分析程序。*/

}

void yyerror(char* s) {
    printf("\nExpression is invalid\n");
    system("pause");
}