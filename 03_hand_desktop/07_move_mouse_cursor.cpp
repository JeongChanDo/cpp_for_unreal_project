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
        std::cout << "Cursor position: x = " << x << " , y = " << y << std::endl;
    }
}

void MoveCursorTo(int x, int y)
{
    SetCursorPos(x, y);
}

int main()
{
    int startX = 80; // ���� x ��ǥ
    int startY = 80; // ���� y ��ǥ
    int duration = 5; // ���α׷� ���� �ð� (��)
    int interval = 500; // ��ǥ ���� ���� (�и���)
    int step = 20; // x ��ǥ ������

    int targetX = startX;
    int targetY = startY;

    // ���� �ð� ���� �ֱ������� x ��ǥ�� �������� ���콺 Ŀ�� �̵�
    for (int i = 0; i < duration * 1000 / interval; i++)
    {
        targetX += step;
        MoveCursorTo(targetX, targetY);
        GetCursorPosition();
        Sleep(interval);
    }

    return 0;
}
