#include<iostream>
using namespace std;

bool isarith = true;
string str;
int i;
void E();
void T();
void A();
void F();
void B();
void transfer(string a,string &b){//�������ַ����е���������ת��Ϊn �����﷨���� 
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
    int main()
    {
        bool flag = false;
        while (flag == false) {
            cout << "�������������䣬��$����" << endl;
            cin >> str;
            if (str[str.length() - 1] == '$' && str.length() != 1) {
                break;
            }
            else {
                cout << "����������������룡" << endl;
            }
        }
        string empty = "";
        transfer(str, empty);
        str = empty;
        i = 0;
        E();
        if (str[i] == '$' && isarith == true)
        {
            cout<<"���Ϸ�"<<endl;
        }
        else
        {
            cout<<"���Ϸ�"<<endl;
        }
        return 0;
    }
void E()
{
    cout<<"E->TA"<<endl;
    T();
    A();

}

void T()
{
    cout << "T->FB" << endl;
    F();
    B();
}
void A()
{
    if (str[i] == '+') {
        i++;
        cout << "A->+TA" << endl;
        T();
        A();
    }
    else if (str[i] == '-')
    {
        cout<<"A->-TA"<<endl;
        i++;
        T();
        A();
    }

}

void F()
{
    if (str[i] == '(')
    {
        i++;
        E();
        if (str[i] == ')')
        {
            i++;
            cout<<"F->(E)"<<endl;
        }
        else
            isarith = false;
    }
    else if (str[i] =='n')
    {
        cout<<"F->num"<<endl;
        i++;
    }
    else
        isarith = false;
}

void B() {

    if (str[i] == '*')
    {
        cout << "B->*FB" << endl;
        i++;
        F();
        B();
    }
    else if (str[i] == '/')
    {
        cout << "B->/FB" << endl;
        i++;
        F();
        B();
    }

}

