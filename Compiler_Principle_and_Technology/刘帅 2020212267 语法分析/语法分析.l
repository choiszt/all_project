
%{  
    #include "yacc.tab.h"
    void yyerror(char *);
	#define YYSTYPE double
	extern YYSTYPE yylval;
%}

digit [0-9]
%option noyywrap /*不会调用yywrap 因为没有文件读写 */


%%   
{digit}+(\.{digit}+)?    {yylval=atof(yytext);return NUMBER;}
[ \t\n]+ {}
.          {return yytext[0];}
%%
