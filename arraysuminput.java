import java.util.*;
class sum1{
    public static void main(String []arg){
        Scanner sc=new Scanner(System.in);
        int n,a=sc.nextInt(),b=sc.nextInt(),z,sum;
        for(int i=a;i<=b;i++)
        {
            n=i/10;
            z=i%10;
            sum=z+n;
            System.out.println(sum);
        }
        
    }
}
