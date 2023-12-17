﻿#include<iostream>
#include<stdio.h>
#include<vector>
#include<string>
#include<fstream>	
using namespace std;
#define buffersize 256
int state;//状态指示
char C;//存放当前读入的字符
string tempstr;
char buffer[buffersize];
int forwardpoint = -1; //字符指针
int rows = 1;	//文件行数
int sum_char = 0; //文件总字数
struct token {
	string mark; //记号
	string name; //属性
	int count;	//出现次数
};

vector<string> keyword = { "auto","break","case","char","const","continue","default","do","double","else","enum","extern",
						   "float","for","goto","if","inline","int","long","register","restrict","return","short","signed",
							"sizeof","string","static","struct","switch","typedef","union","unsigned","void","volatile","while",
							"_Alignas","_Alignof","_Atomic","_Bool","_Complex","_Generic","_Imaginary","_Noreturn","_Static_assert"
							"_Thread_local" };

void get_char() {//从buffer中读入字符
	forwardpoint = (forwardpoint + 1) % buffersize;
	C = buffer[forwardpoint];
}

void cat() {//将字符C连接到nowstr字符串后面
	tempstr.push_back(C);
}

void retract() { //回溯到上一个读入的字符
	forwardpoint = (forwardpoint - 1) % buffersize;
}

bool is_Num(char c) {
	if (c >= '0' && c <= '9')
		return true;
	return false;
}

bool is_letter(char c) {
	if ((c >= 'a' && c <= 'z') || (c <= 'Z' && c >= 'A') || c == '_')
		return true;
	return false;
}

bool is_keyword() {
	for (int i = 0; i < keyword.size(); i++) {
		if (tempstr == keyword[i]) {
			return true;
		}
	}
	return false;
}

bool iseven() {//判断“/”的个数，如果为偶数，说明是注释
	int num = 0;
	int i = tempstr.size() - 2;
	while (tempstr[i] == '\\') {
		num++;
		i--;
	}
	if (num % 2 == 0)
		return true;
	return false;
}
void error(int signal) {
	if (signal == 13)
		cout << "读入了无法识别的字符"<<endl;
	else if (signal == 3)
		cout << "小数点后没有数字"<<endl;
	else if (signal == 5) {
		cout << "指数后没有出现+-或数字"<<endl;
	}
	else if (signal == 6) {
		cout << "+-后没有出现数字"<<endl;
	}
}
void addToken(string a, string b, vector<token>& tempresult) {//将识别到的token放到result
	int i;
	for (i = 0; i < tempresult.size(); i++)
		if (tempresult[i].mark == a && tempresult[i].name == b) {
			tempresult[i].count++;
			break;
		}
	if (i == tempresult.size()) {
		token t;
		t.mark = a;
		t.name = b;
		t.count = 1;
		tempresult.push_back(t);
	}
}
void trace_back() {//回溯
	retract();
	sum_char--;
	if (C == '\n') {
		rows--;
	}
}
vector<token> analysis(ifstream& f) {
	vector<token> result;
	bool flag = false;
	f.read(buffer, buffersize - 1);
	if (f.gcount() < buffersize - 1) {
		buffer[f.gcount()] = EOF;
	}
	buffer[buffersize - 1] = EOF;
	state = 0;
	while (!flag) {
		get_char();
		if (C == '\n')
			rows++;
		if (C != EOF) {
			sum_char++;
		}
		if (C == EOF && forwardpoint != buffersize - 1)
			flag = true;
		else if (C == EOF && forwardpoint == buffersize - 1) {
			f.read(buffer, buffersize - 1);
			if (f.gcount() < buffersize - 1) {
				buffer[f.gcount()] = EOF;
			}
			continue;
		}
		switch (state) {
		case 0:
			if (is_letter(C)) {
				state = 1;
				cat();
			}
			else if (is_Num(C) && C != '0') {
				state = 2;
				cat();
			}
			else {
				switch (C) {
				case '<':state = 8; break;
				case '>':state = 9; break;
				case':':state = 10; break;
				case'?':addToken("分界符", "?", result); break;
				case'(':addToken("分界符", "(", result); break;
				case')':addToken("分界符", ")", result); break;
				case ',':addToken("分界符", ",", result); break;
				case ';': addToken("分界符", ";", result); break;
				case '{': addToken("分界符", "{", result); break;
				case '}': addToken("分界符", "}", result); break;
				case '[': addToken("分界符", "[", result); break;
				case ']': addToken("分界符", "]", result); break;
				case'/':state = 11; break;
				case '=': state = 12; break;
				case '+': state = 13; break;
				case '-': state = 14; break;
				case '*': state = 15; break;
				case '%': state = 16; break;//加减乘除运算
				case '^':state = 17; break;
				case '|': state = 18; break;
				case '~': state = 19; break;
				case '!': state = 20; break;
				case '&': state = 21; break;
				case '\'':state = 23; cat(); break;
				case '.': state = 24; break;
				case EOF:break;
				}
			}
			break;
		case 1:
			if (is_letter(C) || is_Num(C)) {
				cat();
				state = 1;
			}
			else {
				trace_back();
				state = 0;
				if (is_keyword()) {
					addToken("keyword", tempstr, result);
				}
				else
					addToken("id", tempstr, result);
				tempstr.clear();
			}
			break;
		case 2:
			if (is_Num(C)) {
				state = 2;
				cat();
			}
			else {
				switch (C) {
				case '.':cat(); state = 3; break;
				case 'E':cat(); state = 5; break;
				default:
					trace_back();
					state = 0;
					addToken("整数", tempstr, result);
					tempstr.clear();
					break;
				}
			}
			break;
		case 3:
			if (is_Num(C)) {
				cat();
				state = 4;
			}
			else {
				trace_back();
				tempstr.push_back('0');
				addToken("浮点数", tempstr, result);
				state = 0;
				tempstr.clear();
			}
			break;
		case 4:
			if (is_Num(C)) {
				cat();
				state = 4;
			}
			else if (C == 'E') {
				state = 5;
				cat();
			}
			else {
				trace_back();
				state = 0;
				addToken("浮点数", tempstr, result);
				tempstr.clear();
			}
			break;
		case 5:
			if (is_Num(C)) {
				state = 7;
				cat();
			}
			else if (C == '+' || C == '-') {
				cat();
				state = 6;
			}
			else {
				trace_back();
				cout << "第" << rows << "行出现错误"<<"       错误类型：";
				error(5);
				state = 0;
				tempstr.clear();
			}
			break;
		case 6:
			if (is_Num(C)) {
				cat();
				state = 7;
			}
			else {
				trace_back();
				cout << "第" << rows << "行出现错误" << "       错误类型：";

				error(6);
				state = 0;
				tempstr.clear();
			}
			break;
		case 7:
			if (is_Num(C)) {
				cat();
				state = 7;
			}
			else {
				trace_back();
				state = 0;
				addToken("指数", tempstr, result);
				tempstr.clear();
			}
			break;
		case 8:
			if (C == '=') {
				addToken("关系符", "<=", result);
				state = 0;
			}
			else if (C == '<') {
				addToken("位操作符", "<<", result);
				state = 0;
			}
			else {
				addToken("关系符", "<", result);
				trace_back();
				state = 0;
			}
			break;
		case 9:
			if (C == '='){
				addToken("关系符",">=",result);
				state = 0;
			}
			else if (C == '>') {
				addToken("位操作符", ">>", result);
				state = 0;
			}
			else {
				addToken("关系符", ">", result);
				trace_back();
				state = 0;
			}
			break;
		case 10:
			if (C == '=') {
				addToken("关系符", ":=", result);
				state = 0;
			}
			else {
				addToken("分界符", ":", result); break;
				trace_back();
				state = 0;
			}
		case 11:
			switch (C) {
			case'/':
				state = 27;
				break;
			case'*':
				state = 25;
				break;
			case'=':
				addToken("赋值运算符", "/=", result);
				state = 0;
				break;
			default:
				addToken("算数运算符", "/", result);
				trace_back();
				state = 0;
				break;
			}
			break;
		case 25:
			if (C == '*')
				state = 26;
			else {
				state = 25;
				cat();

			}
			break;
		case 26:
			if (C == '/') {
				trace_back();
				state = 0;
				addToken("字符串", tempstr, result);
				tempstr.clear();
			}
			else {
				state = 25;
				cat();
			}
			break;
		case 27:
			if (C == '\n')
				state = 0;
			else
				state = 27;
			break;
		case 12:
			if (C == '=') {
				addToken("关系符", "==", result);
				state = 0;
			}
			else {
				addToken("赋值运算符", "=", result);
				state = 0;
				trace_back();
			}
			break;
		case 13:
			if (C == '=') {
				addToken("赋值运算符", "+=", result);
				state = 0;
			}
			else if (C == '+') {
				addToken("算数运算符", "++", result);
			}
			else {
				addToken("算数运算符", "+", result);
				state = 0;
				trace_back();
			}
			break;
		case 14:
			if (C == '=') {
				addToken("赋值运算符", "-=", result);
				state = 0;
			}
			else if (C == '-') {
				addToken("算术运算符", "--", result);
				state = 0;
			}
			else if (C == '>') {
				addToken("特殊操作符", "->", result);
			}
			else {
				addToken("算术运算符", "-", result);
				state = 0;
				trace_back();
			}
			break;
		case 15:
			if (C == '=') {
				addToken("赋值运算符", "*=", result);
				state = 0;
			}
			else {
				addToken("算术运算符", "*", result);
				state = 0;
				trace_back();
			}
			break;
		case 16:
			if (C == '=') {
				addToken("赋值运算符", "%=", result);
				state = 0;
			}
			else {
				addToken("算术运算符", "%", result);
				state = 0;
				trace_back();
			}
			break;
		case 17:
			if (C == '=') {
				addToken("赋值运算符", "^=", result);
				state = 0;
			}
			else {
				addToken("位运算符", "^", result);
				state = 0;
				trace_back();
			}
			break;
		case 18:
			if (C == '=') {
				addToken("赋值运算符", "|=", result);
				state = 0;
			}
			else if (C == '|') {
				addToken("逻辑运算符", "||", result);
				state = 0;
			}
			else {
				addToken("位运算符","|", result);
				state = 0;
				trace_back();
			}
			break;
		case 19:
			if (C == '=') {
				addToken("赋值运算符", "~=", result);
				state = 0;
			}
			else {
				addToken("位运算符", "~", result);
				state = 0;
				trace_back();
			}
			break;
		case 20:
			if (C == '=') {
				addToken("关系符", "!=", result);
				state = 0;
			}
			else {
				addToken("逻辑运算符", "!", result);
				state = 0;
				trace_back();
			}
			break;
		case 21:
			if (C == '&') {
				addToken("逻辑运算符", "&&", result);
				state = 0;
			}
			else {
				addToken("特殊操作符", "&", result);
				state=0;
				trace_back();
			}
			break;
		case 22://由于读入时中文会显示乱码 将此状态隐去
			if (C == '"') {
				cat();
				if (iseven()) {
					addToken("字符串", tempstr, result);
					tempstr.clear();
					state = 0;
				}
				else state = 22;
			}
			else {
				cat(); 
				state = 22;
			}
			break;
		case 23:
			if (C == '\'') {
				cat();
				if (iseven()) {
					addToken("字符", tempstr, result);
					tempstr.clear();
					state = 0;
				}
				else state = 23;
			}
			else {
				cat();
				state = 23;
			}
			break;
		case 24:
			if (isdigit(C)) {
				tempstr.push_back('0');
				tempstr.push_back('.');
				cat();
				state = 4;
			}
			else {
				addToken("特殊操作符", ".", result);
				trace_back();
				state = 0;
			}
			break;
		default:
			break;
		}
	}
	return result;
}

void output(vector<token> result) {
	int count_keyword = 0;
	int count_id = 0;
	int count_int = 0;
	int count_float = 0;
	int count_exponent = 0;
	int count_relationalOperator = 0;
	int count_logicOperator = 0;
	int count_bitOperator = 0;
	int count_assignOperator = 0;
	int count_specialOperator = 0;
	int count_arithmeticOperator = 0;
	int count_string = 0;
	int count_char = 0;
	int count_delimeter = 0;
	cout << "记号                " << "属性" << "                          出现次数" << endl;
	for (int i = 0; i < result.size(); i++) {
		int j = 30 - result[i].name.size();
		for (int k = 0; k < j; k++) {
			result[i].name.push_back(' ');
		}
	}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "keyword") {
			cout << result[i].mark <<"   "<<result[i].name << result[i].count << endl;
			count_keyword++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "id") {
			cout << result[i].mark <<"   "<<result[i].name << result[i].count << endl;
			count_id++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "整数") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_int++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "浮点数") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_float++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "指数") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_exponent++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "关系符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_relationalOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "逻辑运算符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_logicOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "位操作符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_bitOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "赋值运算符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_assignOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "特殊操作符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_specialOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "算术运算符") {
			cout << result[i].mark << "   " << result[i].name << result[i].count << endl;
			count_arithmeticOperator++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "字符串") {
			cout << result[i].mark<<"   " << result[i].name << result[i].count << endl;
			count_string++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "字符") {
			cout << result[i].mark << result[i].name << result[i].count << endl;
			count_char++;
		}
	for (int i = 0; i < result.size(); i++)
		if (result[i].mark == "分界符") {
			cout << result[i].mark << result[i].name << result[i].count << endl;
			count_delimeter++;
		}
	if (count_keyword > 0)
		cout << "keywords字数:  " << count_keyword << endl;
	if (count_id > 0)
		cout << "id字数:" << count_id << endl;
	if (count_int > 0)
		cout << "整数个数:" << count_int << endl;
	if (count_float > 0)
		cout << "浮点数个数:" << count_float << endl;
	if (count_exponent > 0)
		cout << "指数个数:" << count_exponent << endl;
	if (count_relationalOperator > 0)
		cout << "关系符个数:" << count_relationalOperator << endl;
	if (count_logicOperator > 0)
		cout << "逻辑运算符个数:" << count_logicOperator << endl;
	if (count_bitOperator > 0)
		cout << "位操作符个数:" << count_bitOperator << endl;
	if (count_assignOperator > 0)
		cout << "赋值运算符个数:" << count_assignOperator << endl;
	if (count_specialOperator > 0)
		cout << "特殊操作符个数:" << count_specialOperator << endl;
	if (count_arithmeticOperator > 0)
		cout << "算术运算符个数:" << count_arithmeticOperator << endl;
	if (count_string > 0)
		cout << "字符串个数:" << count_string << endl;
	if (count_char > 0)
		cout << "字符个数:" << count_char << endl;
	if (count_delimeter > 0)
		cout << "分界符个数:" << count_delimeter << endl;

	cout << "行数:" << rows << endl;
	cout << "字数:" << sum_char << endl;
}
//输出结果

int main(void) {
	ifstream fs;
	fs.open("test.txt", ios::in);
	if (fs.is_open() == false)
		exit(0);
	vector<token> result = analysis(fs);
	fs.close();
	output(result);
}