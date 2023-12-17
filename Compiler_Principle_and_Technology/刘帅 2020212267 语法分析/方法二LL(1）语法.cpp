#include<iostream>
#include<stack>
#include<iomanip>
using namespace std;


char grammar[10][8];//产生式集
char terminal[9];//终结符
char noterminal[6];//非终结符
char first[5][8];
char follow[5][8];
int table[5][8];//预测分析表
string str; //输入字符串
stack<char> s,tempstack;
void showstack(stack<char>& tempstack);
void showstr(int ip);
void printgrammar(int flag) {
	if (flag == 0) cout << "E->TA";
	else if (flag == 1) cout << "A->+TA";
	else if (flag == 2) cout << "A->-TA" ;
	else if (flag == 3) cout << "A->ε" ;
	else if (flag == 4) cout << "T->FB" ;
	else if (flag == 5) cout << "B->*FB" ;
	else if (flag == 6) cout << "B->/FB";
	else if (flag == 7) cout << "B->ε";
	else if (flag == 8) cout << "F->(E)";
	else if (flag == 9) cout << "F->num" ;
	else if (flag == -1) cout << "error";
	else if (flag == -2) cout << "synch";
}
void init() {
	s.push('$');
	s.push('E');
	strcpy_s(grammar[0], "E#TA#");
	strcpy_s(grammar[1], "A#+TA#");
	strcpy_s(grammar[2], "A#-TA#");
	strcpy_s(grammar[3], "A#e#");
	strcpy_s(grammar[4], "T#FB#");
	strcpy_s(grammar[5], "B#*FB#");  
	strcpy_s(grammar[6], "B#/FB#");  
	strcpy_s(grammar[7], "B#e#"); 
	strcpy_s(grammar[8], "F#(E)#");  
	strcpy_s(grammar[9], "F#n#");

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 8; j++) {
			table[i][j] = -1;
		}
	}
	strcpy_s(terminal, "+-*/()n$");  //初始化终结符表  
	strcpy_s(noterminal, "EATBF");  //初始化非终结符表   
	strcpy_s(first[0], "(n#");
	strcpy_s(first[1], "+-e#");
	strcpy_s(first[2], "(n#");
	strcpy_s(first[3], "*/e#");
	strcpy_s(first[4], "(n#");

	strcpy_s(follow[0], ")$#");
	strcpy_s(follow[1], ")$#");
	strcpy_s(follow[2], "+-)$#");
	strcpy_s(follow[3], "+-)$#");
	strcpy_s(follow[4], "+-*/)$#");
}
int getnoterminal_index(char ch) {//找出非终结符下标的index
	for (int i = 0; i < 5; i++) {
		if (ch == noterminal[i])
			return i;
	}
	return -1;
}
int getterminal_index(char ch) {
	for (int i = 0; i < 8; i++) {
		if (ch == terminal[i])
			return i;
	}
	return -1;
}
bool infirst(char left, char ch) {//查看ch是否在非终结符的first集中
	int index = getnoterminal_index(left);
	for (int i = 0; first[index][i] != '#'; i++) {
		if (first[index][i]==ch)
			return true;
	}
	return false;

}
void maketable() {
	for (int i = 0; i < 10; i++) {//对每条产生式遍历
		char left = grammar[i][0]; //产生式左部
		char ch = grammar[i][2];//右边第一个符号 终极符or非终结符or空
		if (ch == 'E' || ch == 'T' || ch == 'A' || ch == 'B' || ch == 'F') {
			for(int j=0;j<8;j++){
				if (infirst(ch, terminal[j])) {
					int index1 = getnoterminal_index(left);
					table[index1][j] = i;
					// cout << left << terminal[j] << i<<endl;
				}
			}
		}
		else if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '(' || ch == ')' || ch == 'n' || ch == '$') {
			int index1 = getnoterminal_index(left);
			int index2 = getterminal_index(ch);
			table[index1][index2] = i; //如果是终结符就直接把产生式放进表里
			//cout << left << terminal[index2] << i << endl;
		}
		else if (ch == 'e') {
			int index1 = getnoterminal_index(left);
			for (int j = 0; follow[index1][j] != '#'; j++) {//从他的follow集里面找对应
				int index2 = getterminal_index(follow[index1][j]);
				table[index1][index2] = i;
				//cout << left << terminal[index2] << i << endl;
			}
		}
	}
	for (int i = 0; i < 5; i++) {
		for (int j = 0; follow[i][j] != '#'; j++) {
			int index1 = getterminal_index(follow[i][j]);
			if (table[i][index1] == -1) {
				table[i][index1] = -2;//如果此时表里是空白，则将follow集里的元素对应的表替换为synch
			}
		}
	}
}
void showtable() {
	for (int i = 0; i < 72; i++) {
		cout << '-';
	}cout << endl;
	for (int i = 0; i < 8; i++) {
		cout <<setw(8)<<terminal[i]<<'|';
	}
	cout << endl;
	for (int i = 0; i < 5; i++) {
		for (int i = 0; i < 72; i++) {
			cout << '-';
		}cout << endl;
		cout << noterminal[i];
		for (int j = 0; j < 8; j++) {
			cout << setw(8);
			printgrammar(table[i][j]);
			cout << '|';
		}
		cout << endl;
	}
	for (int i = 0; i < 72; i++) {
		cout << '-';
	}

}
bool isterminal(char ch) {
	if (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '(' || ch == ')' || ch == 'n' || ch == '$') {
		return true;
	}return false;
}
bool isnotterminal(char ch) {
	if (ch == 'E' || ch == 'T' || ch == 'A' || ch == 'B' || ch == 'F') {
		return true;
	}return false;
}
void transfer(string a,string &b){//将输入字符串中的所有数字转换为n 方便语法分析 
	int length=a.length();
	for(int i=0;i<length;i++){
		if('0'<=a[i]&&a[i]<='9'){
			while('0'<=a[i]&&a[i]<='9')
				i++;
			if(a[i]=='.'){
				i++;
				while('0'<=a[i]&&a[i]<='9')
					i++;
			}
			b+="n";
			i--;
			continue;
		}
		else
			b.push_back(a[i]);
	}

}
void ll1analysis() {
	int ip = 0;//输入指针
	string b = "";
	transfer(str, b);
	str= b;
	//cout << str<<endl;
	int step = 0;
	while (!s.empty() || ip < str.length()) {
		step++;
		cout << "top=" << s.top()<<endl;
		if(s.top()!='$') cout << "step" << step << "  ";
		if (isterminal(s.top())) {//如果栈顶是终极符，则和str进行匹配
			if (s.top() == str[ip])
			{
				s.pop();
				ip++;
			}
			else
			{
				s.pop();
				ip++;
			}
		}
		else//当栈顶是非终结符
		{
			int index1 = getnoterminal_index(s.top());
			int index2 = getterminal_index(str[ip]);
			if (table[index1][index2] != -1 && table[index1][index2] != -2)
			{
				s.pop();
				//cout << "index1=" << index1 << "index2=" << index2 << endl;
				int index = table[index1][index2];
				if (grammar[index][2] != 'e')
				{
					int j = 0;
					for (j = 2; grammar[index][j] != '#'; j++);
					for (j--; j >= 2; j--)//将右部倒序写入stack
					{
						s.push(grammar[index][j]);
						tempstack = s;
					}
					printgrammar(index);
					cout << "  ";
				}
				else {
					printgrammar(index); cout << "  ";
				}

			}
			else if (table[index1][index2] == -1)//error
			{
				ip++;
				cout << "输入字符错误，跳过" <<" ";
			}
			else if (table[index1][index2] == -2) {
					s.pop();
					cout << "弹栈" << " ";
					if (s.empty()) break;
				
			}
		}
		tempstack = s;
		showstack(tempstack);
		cout << "  ";
		showstr(ip);
		cout << endl;
	}
}
void showstack(stack<char>&tempstack) {
	while (!tempstack.empty()) {
		cout<<tempstack.top();
		tempstack.pop();
	}
}
void showstr(int ip) {
	for (ip; ip < str.length(); ip++) {
		cout << str[ip];
	}
}
int main() {
	init();
	maketable();
	showtable();
	cout << endl;
	bool flag=false;
	while (flag == false) {
		cout << "请输入待分析语句，用$结束"<<endl;
		cin >> str;
		if (str[str.length() - 1] == '$' &&str.length()!=1) {
			break;
		}
		else {
			cout<<"输入错误，请重新输入！"<<endl;
		}
	}
	ll1analysis();
	tempstack = s;
}