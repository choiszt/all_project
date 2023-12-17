#include<iostream>
#include<vector>
#include<string>
#include<map>
#include<stack>
using namespace std;
struct gram {
	int num=0;
	string out="";
	char in = 'n';
};

struct node {
	char type='n';
	int num=0;
};
node action[16][8];			//action表
int goTo[16][3];			//goto表
gram grammar[9];
map<char, int> terminal = {
	{'+',0},
	{'-',1},
	{'*',2},
	{'/',3},
	{'(',4},
	{')',5},
	{'n',6},
	{'$',7},
};
map<char, int> nonterminal = {
	{'E',0},
	{'T',1},
	{'F',2},
};
void init() {

	grammar[0] = { 1,"S->E",'S'};
	grammar[1] = { 3,"E->E+T",'E'};
	grammar[2] = { 3,"E->E-T",'E'};
	grammar[3] = { 1,"E->T",'E'};
	grammar[4] = { 3, "T->T*F",'T'};
	grammar[5] = { 3,"T->T/F",'T'};
	grammar[6] = { 1,"T->F",'T'};
	grammar[7] = { 3,"F->(E)",'F'};
	grammar[8] = { 1,"F->num",'F'};
	action[0][terminal['(']] = { 'S',4 };
	action[0][terminal['n']] = { 'S',5 };
	goTo[0][nonterminal['E']] = 1;
	goTo[0][nonterminal['T']] = 2;
	goTo[0][nonterminal['F']] = 3;
	action[1][terminal['+']] = { 'S',6 };
	action[1][terminal['-']] = { 'S',7 };
	action[0][7].type = 'F';
	action[1][7].type = 'F';
	action[2][terminal['+']] = { 'R',3 };
	action[2][terminal['-']] = { 'R',3 };
	action[2][terminal[')']] = { 'R',3 };
	action[2][terminal['$']] = { 'R',3 };
	action[2][terminal['*']] = { 'S',8 };
	action[2][terminal['/']] = { 'S',9 };
	action[3][terminal['+']] = { 'R',6 };
	action[3][terminal['-']] = { 'R',6 };
	action[3][terminal['*']] = { 'R',6 };
	action[3][terminal['/']] = { 'R',6 };
	action[3][terminal[')']] = { 'R',6 };
	action[3][terminal['$']] = { 'R',6 };
	action[4][terminal['(']] = { 'S',4 };
	action[4][terminal['n']] = { 'S',5 };
	goTo[4][nonterminal['E']] = 10;
	goTo[4][nonterminal['T']] = 2;
	goTo[4][nonterminal['F']] = 3;
	action[5][terminal['+']] = { 'R',8 };
	action[5][terminal['-']] = { 'R',8 };
	action[5][terminal['*']] = { 'R',8 };
	action[5][terminal['/']] = { 'R',8 };
	action[5][terminal[')']] = { 'R',8 };
	action[5][terminal['$']] = { 'R',8 };
	action[6][terminal['(']] = { 'S',4 };
	action[6][terminal['n']] = { 'S',5 };
	goTo[6][nonterminal['T']] = 11;
	goTo[6][nonterminal['F']] = 3;
	action[7][terminal['(']] = { 'S',4 };
	action[7][terminal['n']] = { 'S',5 };
	goTo[7][nonterminal['T']] = 12;
	goTo[7][nonterminal['F']] = 3;
	action[8][terminal['(']] = { 'S',4 };
	action[8][terminal['n']] = { 'S',5 };
	goTo[8][nonterminal['F']] = 13;
	action[9][terminal['(']] = { 'S',4 };
	action[9][terminal['n']] = { 'S',5 };
	goTo[9][nonterminal['F']] = 14;
	action[10][terminal['+']] = { 'S',6 };
	action[10][terminal['-']] = { 'S',7 };
	action[10][terminal[')']] = { 'S',15 };
	action[11][terminal['+']] = { 'R',1 };
	action[11][terminal['-']] = { 'R',1 };
	action[11][terminal['*']] = { 'S',8 };
	action[11][terminal['/']] = { 'S',9 };
	action[11][terminal[')']] = { 'R',1 };
	action[11][terminal['$']] = { 'R',1 };
	action[12][terminal['+']] = { 'R',2 };
	action[12][terminal['-']] = { 'R',2 };
	action[12][terminal['*']] = { 'S',8 };
	action[12][terminal['/']] = { 'S',9 };
	action[12][terminal[')']] = { 'R',2 };
	action[12][terminal['$']] = { 'R',2 };
	action[13][terminal['+']] = { 'R',4 };
	action[13][terminal['-']] = { 'R',4 };
	action[13][terminal['*']] = { 'R',4 };
	action[13][terminal['/']] = { 'R',4 };
	action[13][terminal[')']] = { 'R',4 };
	action[13][terminal['$']] = { 'R',4 };
	action[14][terminal['+']] = { 'R',5 };
	action[14][terminal['-']] = { 'R',5 };
	action[14][terminal['*']] = { 'R',5 };
	action[14][terminal['/']] = { 'R',5 };
	action[14][terminal[')']] = { 'R',5 };
	action[14][terminal['$']] = { 'R',5 };
	action[15][terminal['+']] = { 'R',7 };
	action[15][terminal['-']] = { 'R',7 };
	action[15][terminal['*']] = { 'R',7 };
	action[15][terminal['/']] = { 'R',7 };
	action[15][terminal[')']] = { 'R',7 };
	action[15][terminal['$']] = { 'R',7 };
}
void showstack(stack<int>& tempstack) {
	while (!tempstack.empty()) {
		cout << tempstack.top();
		tempstack.pop();
	}
}
void showstack2(stack<char>& tempstack) {
	while (!tempstack.empty()) {
		cout << tempstack.top();
		tempstack.pop();
	}
}
void analysis(string s) {
	int ptr = 0;//字符串指针
	stack<int> state;//状态栈
	state.push(0);
	stack<char>symbol;//目前字符串的栈
	while (true) {
		if (action[state.top()][terminal[s[ptr]]].type == 'F') {
			cout << "识别成功！";
			break;//识别到ACC代表分析结束
		}
		else if (action[state.top()][terminal[s[ptr]]].type =='S') {

			state.push(action[state.top()][terminal[s[ptr]]].num); //状态栈
			symbol.push(s[ptr]);
			ptr++;
		}
		else if (action[state.top()][terminal[s[ptr]]].type == 'R') {
			gram temp = grammar[(action[state.top()][terminal[s[ptr]]].num)];
			cout<<temp.out<<endl;
			int count = temp.num;
			for (int i = 0; i < count; i++) {
				symbol.pop();
			}
			state.pop();
			symbol.push(temp.in);
			state.push(goTo[state.top()][nonterminal[symbol.top()]]);

		}
		else {
			cout << "error!!!" << endl;
			break;
		}
	}
	
}
void transfer(string a, string& b) {//将输入字符串中的所有数字转换为n 方便语法分析 
	int length = a.length();
	for (int i = 0; i < length; i++) {
		if ('0' <= a[i] && a[i] <= '9') {
			while ('0' <= a[i] && a[i] <= '9')
				i++;
			if (a[i] == '.') {
				i++;
				while ('0' <= a[i] && a[i] <= '9')
					i++;
			}
			b += "n";
			i--;
			continue;
		}
		else
			b.push_back(a[i]);
	}
}
int main(void) {
	init();
	string str;
	bool flag = false;
	while (flag == false) {
		cout << "请输入待分析语句，用$结束" << endl;
		cin >> str;
		if (str[str.length() - 1] == '$' && str.length() != 1) {
			break;
		}
		else {
			cout << "输入错误，请重新输入！" << endl;
		}
	}
	string empty = "";
	transfer(str, empty);
	str = empty;
	analysis(str);
	return 0;
}