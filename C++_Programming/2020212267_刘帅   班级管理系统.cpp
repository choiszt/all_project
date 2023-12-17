#include<iostream>
#include<cstdio>
#include<iomanip>
#include<cstdlib>
#include<algorithm>
#include<fstream>
#include<string>
#include<fstream>
#include<time.h>
#include<windows.h>
#include<cstring>
using namespace std;
int readin();
int shendu();
int menu();
int numpeople();
int studentlist();
int search();
int personalinfo();
int namejudge();
int numberjudge();
int getnumber();
int findaddress();
int luru();
int roommate();
int laoxiang();
int tongban();
int editclass();
int editnumber();
int save();
int xiugai();
int xingzuo();
int sortage();
int zongrenshu;
int teding = 0;//namejudge里的m，具体内存中的第m个人
int bencishururenshu;//一次输入的人数（在readin函数中统一txt和内存使用）
int stn;//the number of student
struct student {
	char name[100];//姓名
	char xuehao[100];//学号
	char banji[100];//班级
	char number[100];//班内序号
	char sex[100];//性别
	char year[100];//生日（年）
	char month[100];//生日（月）
	char home[100];//家乡
	char sushe[100];//宿舍
}; 
student stu[1000];
//student* parr = new student[1000];
//student stu1 = { "刘帅",2020212267,10,16,"男",2001,9,"北京" };
int menu() {
	system("color 0E");
		loop:char choice;
		cout << "*******************************************************" << endl;
		cout << "********* 欢迎使用班级管理系统！（主菜单）*************" << endl;
		cout << "************* 0.退出管理程序***************************" << endl;
		cout << "************* 1.统计班级人数***************************" << endl;
		cout << "************* 2.班级学生转入&转出**********************" << endl;
		cout << "************* 3.查找学生信息【找老乡&找星座&找舍友】***" << endl;
		cout << "************* 4.修改学生信息***************************" << endl;
		cout << "************* 5.年龄排序*******************************" << endl;
		cout << "*******************************************************" << endl;
		cout << "*******************************************************" << endl;
		cout << endl;
		cin >> choice;
		switch (choice) {
		case '0':break;
		case '1': numpeople(); break;
		case '2': readin(); break;
		case '3': studentlist(); break;
		case'4':personalinfo(); break;
		case'5':sortage(); break;
		default:cout << "无效操作！请重新输入"; Sleep(500); system("cls"); goto loop;
		}
	return 0;
}

int numpeople() {
	luru();
	char CLASS[100];
	char option;
	int k=0;
	cout << "班级人数统计" << endl << "请输入班级" << endl;
	cin >> CLASS;
	for (int i = 0; i < getnumber(); i++) {
		if (!strcmp(stu[i].banji,CLASS) ){
			k++;
		}
	}
	cout << CLASS << "班共有" << k << "人"<<endl;
	if (k != 0) {
		cout << "他们是 ";
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[i].banji,CLASS)) {
				cout << stu[i].name << "  ";
			}
		}k = 0; cout << endl;
	}
	cout << "继续查找请按1      " << "       返回主菜单请按0"<<endl;//添加针对姓名的学生信息查找
	cin >> option;
	if (option == '1') numpeople();
	else {
		system("cls"); menu();
	}
	return 0;
}
int readin() {
	int  stusum = getnumber();//stusum是txt中所含人数；
	int& p = stusum;
	int yes;
	luru();//把txt的数据录入到内存，保持序号
	cout << "添加学生请按1" << "   返回主菜单请按0" << endl;
	cin >> yes;
	int shuru = 0;
	if (yes == 1) {
		for (p; p < 1000; p++) {
			if (yes == 1) {//此时检测到txt中一共有多少人数，然后在人数后面追加人数
				cout << "请输入" << "姓名  " << "学号  " << "班级  " << "班内序号  " << "性别  " << "生日（年）" << "生日（月）" << "家乡 " << "宿舍" << endl;
				cin >> stu[p].name >> stu[p].xuehao >> stu[p].banji >> stu[p].number >> stu[p].sex >> stu[p].year >> stu[p].month >> stu[p].home >> stu[p].sushe;
				if (stu[p].name == 0 || stu[p].xuehao == 0 || stu[p].banji == 0 || stu[p].number == 0 ||
					stu[p].sex == 0 || stu[p].year == 0 || stu[p].month == 0 || stu[p].home == 0 || stu[p].sushe == 0) {
					cout << "录入出错！请重新录入" << " " << "重新录入请按1  返回主菜单请按0" << endl;
					cin.clear();
					cin.ignore(8912, '\n');
					cin.sync();
					cin >> yes;
					p--;

				}
				else {
					cout << "信息录入成功!!" << endl << "继续录入请按1" << " " << "退出录入请按0" << endl;
					shuru++;
					bencishururenshu = shuru;
					save();
					getnumber();//更新txt中所含人数


					cin >> yes;
				}
			}
			else {
				cout << "信息录入结束！";
				shuru = 0;
				cout << "点击任意键以返回主菜单";
				system("pause");
				system("cls");
				menu();
				break;
			}
		}
	}
	else { system("cls"); menu(); }
	//student* pa = parr;
	return 0;
}
int getnumber() {//txt中所含的人数
	ifstream ifs;
	ifs.open("manager.txt", ios::in);
	char name[100];//姓名
	char xuehao[100];//学号
	char banji[100];//班级
	char number[100];//班内序号
	char sex[100];//性别
	char year[100];//生日（年）
	char month[100];//生日（月）
	char home[100];//家乡
	char sushe[100];//宿舍
	int num = 0;
	int i = 0;
	while (ifs >> name&& ifs >> xuehao && ifs >> banji&& ifs >> number && ifs >> sex&& ifs >> year && ifs >> month && ifs >> home&&ifs>>sushe) {
		num++;
	}
	ifs.close();
	return num;
}
int luru() {
	ifstream ifs;
	ifs.open("manager.txt", ios::in);
	char name[100];//姓名
	char xuehao[100];//学号
	char banji[100];//班级
	char number[100];//班内序号
	char sex[100];//性别
	char year[100];//生日（年）
	char month[100];//生日（月）
	char home[100];//家乡
	char sushe[100];//宿舍
	int num = 0;
	int i = 0;
	while (ifs >> name && ifs >> xuehao && ifs >> banji && ifs >> number && ifs >> sex && ifs >> year && ifs >> month && ifs >> home&&ifs>>sushe) //每行读入
	{
		strcpy_s(stu[i].name,name);
		strcpy_s(stu[i].xuehao, xuehao);
		strcpy_s(stu[i].banji, banji);
		strcpy_s(stu[i].number, number);
		strcpy_s(stu[i].sex, sex);
		strcpy_s(stu[i].year, year);
		strcpy_s(stu[i].month, month);
		strcpy_s(stu[i].home, home);
		strcpy_s(stu[i].sushe, sushe);
		i++;
	}
	ifs.close();
	return 0;
}
int save() {
	ofstream ofs;//创建流对象（将文件读出到txt）
	ofs.open("manager.txt", ios::app);//打开txt文件()   app:将数据读入文件末尾
	int sum = getnumber();
	//cout << sum << "FLAG" << endl;
	for (int i = 0; i <bencishururenshu ; i++) {//bencishururenshu是某次输入的人数
		ofs << stu[sum].name << " " << stu[sum].xuehao << " " << stu[sum].banji << " " << stu[sum].number << " " << stu[sum].sex << " " << stu[sum].year << " " << stu[sum].month << " " << stu[sum].home<<" "<<stu[sum].sushe << endl;
		sum++;
	}//将内存数据读入txt文件
	//cout << "getnumber的值是" << getnumber()<<endl;
	return 0;
}

int search() {
	luru();
	int button3;
	cout << "总学生个数为" << getnumber() << endl;
	for (int i = 0; i < getnumber(); i++) {
			cout << stu[i].name << " " << stu[i].xuehao << " " << stu[i].banji << " " << stu[i].number
				<< " " << stu[i].sex << " " << stu[i].year << " " << stu[i].month << " " << stu[i].home<<" "<<stu[i].sushe << endl;
		}
	cout << "按任意键返回上级菜单"<<endl;
	cin >>button3;
	studentlist();
	return NULL;
}    


int studentlist() {
	luru();
	char option = 0;
	cout << "        学生信息查找 " << endl;
	cout << "************* 1.学生个人信息" << endl;
	cout << "************* 2.全体学生详细信息" << endl;
	cout << "************* 3.由地址找学生" << endl;
	cout << "************* 4.返回主菜单" << endl;
	cin >> option;
	switch (option) {
	case '1': personalinfo(); break;
	case '2': search(); break;
	case'3':findaddress(); break;
	default:system("cls"); menu(); break;
	}
	return 0;
}
int sortage() {
	luru();
	int p=0;
	string a;//将内存数据复制到a；
	string b[100];//名字
	string c[100];//年龄
	int d[100] = {0};
	for (int i = 0; i < getnumber(); i++) {b[i] =string( stu[i].name);c[i] = string(stu[i].year);}
	for (int i = 0; i < getnumber(); i++)
	{if (c[i] == "2001") d[i] = 19;
		else if (c[i] == "2002")d[i] = 18;
		else if (c[i] == "2003")d[i]=17;}
	for (int i = 0; i < getnumber()-1; i++) {
		for (int i = 0; i < getnumber() - 1; i++) {
			if (stu[i].year > stu[i + 1].year) {
				swap(b[i], b[i + 1]);
				swap(c[i], c[i + 1]);
			}	}}
	for (int i = 0; i < getnumber(); i++) {

		cout << b[i] << " " << d[i]<<"岁"<<endl;
	}
	cout << "按任意键返回主菜单";
	cin >>p;
	system("cls"); menu();
	return 0;
}

int findaddress() {
	char address[100] = {0};
	char option6='0';
	cout<<"请输入省份(两个字，如“北京”、“上海” 或 三个字，如“内蒙古”)"<<endl;
	cin >> address;
	int k = 0;
	for (int i = 0; i < getnumber(); i++) {
		if (!strcmp(stu[i].home, address)) {
			k++;
		}
	}
	if (k != 0) {
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[i].home, address)) {
				cout << stu[i].name<<" ";
			}
		}
			cout << "是" << address << "人" << endl;
		}
		else cout << "未找到" << address << "人"<<endl;
		cout << "************* 1.继续查找" << endl;
		cout << "************* 2.返回上级菜单" << endl;
		cout << "************* 3.返回主菜单" << endl;
		loop1:cin >> option6;
		switch (option6) {
		case'1':findaddress(); break;
		case'2':studentlist(); break;
		case'3':system("cls"); menu(); break;
		default:cout << "无效操作，请重新输入！"; goto loop1;
		}
	return 0;
}
int personalinfo() {
	int option2;
	cout << "************* 1.学生个人信息查找（按姓名) " << endl;
	cout << "************* 2.学生个人信息查找（按学号)" << endl;
	cout << "************* 3.返回上级菜单" << endl;
	cin >> option2;
	switch (option2) {
	case(1):namejudge(); break;
	case(2):numberjudge(); break;
	case(3):studentlist(); break;
	}
	return 0;
}
int xiugai() {
	char button = '0';
	cout << "************* 修改学生信息 " << endl;
	cout << "**************1.修改" << stu[teding].name << "的班级" << endl;
	cout << "**************2.修改" << stu[teding].name << "的学号" << endl;
	cout << "**************3.返回上级菜单" << endl;
	cin >> button;
	switch (button) {
	case'1':editclass(); break;
	case'2':editnumber(); break;
	default:personalinfo(); break;

	}
	return 0;
}
int editclass() {
	//luru();
	cout<< "请输入修改后的班级"<<endl;
	char  a[100];
	string temp1;
	char option7;
	cin >> a;
	temp1=string(stu[teding].banji) ;
	int sum;
	strcpy_s(stu[teding].banji, a);
	cout << "修改成功！" << endl;
		ofstream ofs;//创建流对象（将文件读出到txt）
		ofs.open("manager.txt", ios::out);//打开txt文件()   app:将数据读入文件末尾
		//cout << sum << "FLAG" << endl;
		sum = 0;
		for (int i = 0; i < getnumber(); i++) {//bencishururenshu是某次输入的人数
			ofs << stu[sum].name << " " << stu[sum].xuehao << " " << stu[sum].banji << " " << stu[sum].number << " " << stu[sum].sex << " " << stu[sum].year << " " << stu[sum].month << " " << stu[sum].home << " " << stu[sum].sushe << endl;
			sum++;
		}//将内存数据读入txt文件
		//cout << "getnumber的值是" << getnumber()<<endl;
		luru();
	cout << stu[teding].name << "的班级由" << temp1 << "班" << "修改为" << stu[teding].banji << "班" << endl;
	loop3: cout << "************* 1.继续修改 " << endl;
	cout << "************* 2.返回上级菜单" << endl;
	cin >> option7;
	switch (option7) {
	case'1':xiugai(); break;
	case'2':studentlist(); break;
	default:cout << "无效口令！请重新输入" << endl; goto loop3;
	}
	return 0;
}
int editnumber() {
	luru();
	cout << "请输入修改后的学号" << endl;
	char b[100];
	string temp1;
	char option8;
	cin >> b;
	temp1 = string(stu[teding].xuehao);
	strcpy_s(stu[teding].xuehao, b);
	cout << "修改成功！" << endl;
	cout << stu[teding].name << "的学号由" << temp1<< "修改为" << stu[teding].xuehao<< endl;
loop4: cout << "************* 1.继续修改 " << endl;
	cout << "************* 2.返回上级菜单" << endl;
	cin >> option8;
	switch (option8) {
	case'1':xiugai(); break;
	case'2':studentlist(); break;
	default:cout << "无效口令！请重新输入" << endl; goto loop4;
	}
	return 0;
}

int namejudge() {
	luru();
	char option = 0;
	bool flag = 0;
	char button = 0;
	int m = 0;//库里面的第m个人是所查名字的人
	char name[50];
	do {
		cout << "请输入学生姓名" << endl;
		cin >> name;
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[i].name, name)) {
				flag = 1;
				m = i;
				teding = m;
				break;
			}
			else flag = 0;
		}
		if (flag == 1) {
			cout << "该生的信息为" << endl;
			cout << stu[m].name << " " << stu[m].xuehao << " " << stu[m].banji << " " << stu[m].number
				<< " " << stu[m].sex << " " << stu[m].year << " " << stu[m].month << " " << stu[m].home<<" 1  "<<stu[m].sushe << endl;
			cout << "************* 1.深度查找" << endl;
			cout << "************* 2.查找其他学生(按姓名)" << endl;
			cout << "************* 3.返回上级菜单" << endl;
			loop5:cin >> option;
			switch (option) {
			case('1'):shendu(); break;
			case('2'):namejudge(); break;
			case('3'):personalinfo(); break;
			case('4'):cout << "无效操作，请重新输入"; goto loop5;
			}
		}
		else {
			cout << "无此学生信息！" << endl;
			cout << "************* 1.重新查找" << endl;
			cout << "************* 2.返回上级菜单（学生个人信息）" << endl;
			cout << "************* 3.返回主菜单" << endl;
			cin >> button;
		}
	} while (button == '1');
	if (button == '2') {
		personalinfo();
	}
	else if (button == '3') {
		system("cls");
		menu();
	}
	return 0;
}
int numberjudge() {
	luru();
	char button2 = '0';
	char option2;
	bool flag2 = 0;
	int m1 = 0;
	char xuehao[100] = { 0 };
	do {
		cout << "请输入该学生学号" << endl;
		cin >> xuehao;
		//cout << getnumber() << "FLAG";
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[i].xuehao, xuehao)) {
				flag2 = 1;
				m1 = i;
				teding = m1;
				break;
			}
			else flag2 = 0;
		}if (flag2 == 1) {
			cout << "该生的信息为" << endl;
			cout << stu[m1].name << " " << stu[m1].xuehao << " " << stu[m1].banji << " " << stu[m1].number
				<< " " << stu[m1].sex << " " << stu[m1].year << " " << stu[m1].month << " " << stu[m1].home << stu[m1].sushe << endl;
			cout << "************* 1.深度查找" << endl;
			cout << "************* 2.查找其他学生(按学号)" << endl;
			cout << "************* 3.返回上级菜单" << endl;
			cin >> option2;
			switch (option2) {
			case('1'):shendu(); break;
			case('2'):numberjudge(); break;
			case('3'):personalinfo(); break;
			}
		}
		else {
			cout << "无此学生信息！" << endl;
			cout << "************* 1.重新查找" << endl;
			cout << "************* 2.返回上级菜单（学生个人信息）" << endl;
			cout << "************* 3.返回主菜单" << endl;
			cin >> button2;
		}
	} while (button2 == '1');
	if (button2 == '2') {
		personalinfo();
	}
	else if (button2 == '3') {
		menu();
	}
	return 0;
}
int shendu() {
	char option3='0';
	cout <<"**************1.查找"<< stu[teding].name<<"的宿舍舍友" << endl;
	cout << "************* 2.查找"<<stu[teding].name<<"的老乡" << endl;
	cout << "************* 3.查找"<<stu[teding].name<<"的同班同学"<< endl;
	cout << "**************4.查找" << stu[teding].name << "的星座" << endl;
	cout << "**************5.修改" << stu[teding].name << "的信息" << endl;
	cout << "**************6.返回上级菜单" << endl;
	cin >> option3;
	switch (option3) {
	case('1'):roommate(); break;
	case('2'):laoxiang(); break;
	case('3'):tongban(); break;
	case('4') :xingzuo(); break;
	case('5'):xiugai(); break;
	default:personalinfo(); break;
	}
	return 0;
}
int xingzuo() {
	enum xingzuo { 水瓶座，双鱼座，白羊座，金牛座，双子座，巨蟹座，狮子座，处女座，天秤座，天蝎座，射手座，魔羯座 };
	xingzuo day=(xingzuo)0;
	if ((string)stu[teding].month == "1")  day = (xingzuo)0;
	else if ((string)stu[teding].month == "2")  day = (xingzuo)1;
	else if ((string)stu[teding].month == "3")  day = (xingzuo)2;
	else if ((string)stu[teding].month == "4")  day = (xingzuo)3;
	else if ((string)stu[teding].month == "5")  day = (xingzuo)4;
	else if ((string)stu[teding].month == "6")  day = (xingzuo)5;
	else if ((string)stu[teding].month == "7")  day = (xingzuo)6;
	else if ((string)stu[teding].month == "8")  day = (xingzuo)7;
	else if ((string)stu[teding].month == "9")  day = (xingzuo)8;
	else if ((string)stu[teding].month == "10")  day = (xingzuo)9;
	else if ((string)stu[teding].month == "11")  day = (xingzuo)10;
	else if ((string)stu[teding].month == "12")  day = (xingzuo)11;
	switch (day) {
	case 0:cout << "水瓶座"; break;
	case 1:cout << "双鱼座"; break;
	case 2:cout << "白羊座"; break;
	case 3:cout << "金牛座"; break;
	case 4:cout << "双子座"; break;
	case 5:cout << "巨蟹座"; break;
	case 6:cout << "狮子座"; break;
	case 7:cout << "处女座"; break;
	case 8:cout << "天秤座"; break;
	case 9:cout << "天蝎座"; break;
	case 10:cout << "射手座"; break;
	case 11:cout << "魔羯座"; break;
	}
	char option6;
	cout << endl << "************* 1.继续查找" << stu[teding].name << "同学信息" << endl;
	cout << "************* 2.查找其他同学" << endl;
	cout << "************* 3.返回主菜单" << endl;
	cin >> option6;
	switch (option6) {
	case('1'):shendu(); break;
	case('2'):personalinfo(); break;
	case('3'):system("cls"); menu(); break;
	}
	return 0;
}
int roommate() {
	char option3='0';

	int k = 0;
	for (int i = 0; i < getnumber(); i++) {
		if (!strcmp(stu[teding].sushe, stu[i].sushe) && i != teding) {
			k++;
		}
	}
	if (k != 0) {
		cout << stu[teding].name << "的舍友是：";
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[teding].sushe, stu[i].sushe) && i != teding) {
				cout << stu[i].name << " ";
			}
		}
	}
	else cout << "未找到" << stu[teding].name << "的舍友";
	cout << endl << "************* 1.继续查找" << stu[teding].name << "同学信息" << endl;
	cout << "************* 2.查找其他同学" << endl;
	cout << "************* 3.返回主菜单" << endl;
	cin >> option3;
	switch (option3) {
	case('1'):shendu(); break;
	case('2'):personalinfo(); break;
	case('3'):system("cls"); menu(); break;
	}
		return 0;
}

int laoxiang() {
	int k = 0;
	char option4 = '0';

	for (int i = 0; i < getnumber(); i++) {
		if (!strcmp(stu[teding].home, stu[i].home) && i != teding) k++;
	}
	//cout << k << "flag";
	if (k != 0) {
		cout << stu[teding].name << "的老乡是：";
		for (int i = 0; i < getnumber(); i++) {
			if (!strcmp(stu[teding].home, stu[i].home) && i != teding) {
				cout << stu[i].name << " ";
			}
		}
	}
	else cout << "未找到" << stu[teding].name << "的老乡";
	cout << endl << "************* 1.继续查找" << stu[teding].name << "同学信息" << endl;
	cout << "************* 2.查找其他同学" << endl;
	cout << "************* 3.返回主菜单" << endl;
	cin >> option4;
	switch (option4) {
	case('1'):shendu(); break;
	case('2'):personalinfo(); break;
	case('3'):system("cls"); menu(); break;
	}
	return 0;
}
int tongban() {
	char option5 = '0';
	int k = 0;
	cout << stu[teding].name << "的同班同学是：";
	for (int i = 0; i < getnumber(); i++) {
		if (!strcmp(stu[teding].banji, stu[i].banji) && i != teding) {
			cout << stu[i].name << " ";
		}
	}
	cout << endl << "************* 1.继续查找" << stu[teding].name << "同学信息" << endl;
	cout << "************* 2.查找其他同学" << endl;
	cout << "************* 3.返回主菜单" << endl;
	cin >> option5;
	switch (option5) {
	case('1'):shendu(); break;
	case('2'):personalinfo(); break;
	case('3'):system("cls"); menu(); break;
	}
	return 0;
}

int main() {
	menu();
	/*string str;
	ifstream in("info.txt");
	if (in.is_open()) {
		while (!in.eof()) {
			getline(in, str);
			cout << str << endl;
		}
	}*/
	//student stu[10]; w
	//student* p = &stu1;
	//cout << p->name;
	return 0;
}