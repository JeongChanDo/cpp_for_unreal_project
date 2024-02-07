#include <Windows.h>
#include <iostream>

void GetCursorPosition()
{
    POINT cursorPos;
    if (GetCursorPos(&cursorPos))
    {
        int x = cursorPos.x;
        int y = cursorPos.y;
        // ���콺 Ŀ�� ��ġ�� ����Ͽ� ���ϴ� �۾� ����
        // ��: ��ǥ ���
        std::cout << "Cursor position: x = "<< x << " , y = " << y << std::endl;
    }
}

int main()
{
    int duration = 5; // ���α׷� ���� �ð� (��)
    int interval = 500; // ���콺 Ŀ�� ��ġ ���� ���� (�и���)

    // ���� �ð� ���� �ֱ������� ���콺 Ŀ�� ��ġ ���
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        GetCursorPosition();
        Sleep(interval);
    }

    return 0;
}
